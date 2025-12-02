from itertools import combinations, product, chain
from functools import reduce
from enum import Enum, auto
from collections import defaultdict, Counter
from typing import Dict, Set, Iterator, Tuple, List, Any
from grammar import (
    Expression,
    Variable,
    Sum,
    Number,
    GreaterThan,
    LessThan,
    VariableState,
)
from constraints import EqualityConstraint, InequalityConstraint, PartialConstraint, generate_random_constraints
from utils.assignment_utils import integrate_constraints_and_forget, get_IG_subproblem, remove_solved_variables
from utils.assignment_utils import print_assignments, get_solved_variables, get_sizes_from_culling, get_variable_counts
import numpy as np
import random
import csv
from dataclasses import dataclass, asdict
import math
from pathlib import Path

class Result(Enum):
    ACCEPT = auto()
    REJECT = auto()
    CONTRADICTION = auto()
    NO_MORE_CONSTRAINTS = auto()

class AgentState(Enum):
    INITIAL = auto()
    UPDATE = auto()
    FINISHED = auto()


def reset_constraints(constraints):
    for c in constraints:
        for v in c.variables:
            v.unassign()



def expected_deductions_by_time(assignments, p, t):
    if not assignments:
        raise ValueError("assignments must be non-empty.")
    if not (0 <= p < 1):
        raise ValueError(f"p={p}, must be in [0,1).")
    if t < 0:
        raise ValueError(f"t={t}, must be >= 0.")

    N = len(assignments)
    vars_set = set().union(*assignments)

    counts = get_variable_counts(assignments)

    # Compute r_t = 1 - (1 - p)^t in a numerically stable way
    if p == 0:
        r_t = 0.0
    else:
        log_qt = t * math.log1p(-p)
        r_t = -math.expm1(log_qt)
        
    # Clamp to [0, 1] to avoid tiny negative/overshoot due to rounding
    if r_t < 0.0:
        r_t = 0.0
    elif r_t > 1.0:
        r_t = 1.0

    # Precompute log(r_t) when needed
    log_r_t = None if r_t == 0.0 else math.log(r_t)

    per_var = {}
    for v in vars_set:
        log_terms = []
        for d in v.domain:
            k_vd = counts[v][d]
            if k_vd <= 0:
                continue

            exponent = N - k_vd
            # weight term: k_vd / N
            log_weight = math.log(k_vd / N)

            if exponent == 0:
                # r_t ** 0 = 1 regardless of r_t
                log_terms.append(log_weight)
            elif r_t == 0.0:
                # term is exactly 0 for positive exponent when r_t == 0
                # skip adding since it contributes nothing
                continue
            else:
                # General case: log((k_vd/N) * r_t ** (N - k_vd))
                log_terms.append(log_weight + exponent * log_r_t)

        if not log_terms:
            per_var[v] = 0.0
        else:
            # Stable log-sum-exp
            m = max(log_terms)
            s = sum(math.exp(x - m) for x in log_terms)
            per_var[v] = math.exp(m) * s

    total_expected = sum(per_var.values())
    return total_expected







class SubProblem:
    def __init__(self):
        self.unsimplified_constraints = []
        self.constraints = []
        self.variables = set()
        self.assignments = []
        self.information_loss, self.information_gain = 0, 0
        self.steps = 0
        self.found_contradiction = False

    def copy(self):
        subproblem = SubProblem()
        subproblem.constraints = self.constraints.copy()
        subproblem.unsimplified_constraints = self.unsimplified_constraints.copy()
        subproblem.variables = self.variables.copy()
        subproblem.assignments = self.assignments.copy()

        subproblem.information_loss = self.information_loss
        subproblem.information_gain = self.information_gain
        subproblem.steps = self.steps
        subproblem.found_contradiction = self.found_contradiction
        return subproblem

    def integrate_and_cull(self, constraint, memory_capacity):
        assignments, information_loss = integrate_constraints_and_forget(
            self.assignments.copy(), constraint, memory_capacity=memory_capacity
        )

        self.assignments = assignments.copy()
        self.information_loss += information_loss
        self.information_gain = get_IG_subproblem(self.assignments)

    def add(self, constraint, subset_vs=None, memory_capacity=np.inf):

        if subset_vs is None:
            vs = constraint.get_unassigned()
        else:
            vs = subset_vs
        partial_constraint = PartialConstraint(constraint, vs)
        self.constraints.append(partial_constraint)
        self.unsimplified_constraints.append(constraint)
        self.variables.update(vs)
        self.integrate_and_cull(partial_constraint, memory_capacity)
        return self

    def increment_step(self):
        self.steps += 1

    def V(self, IL_max):
        if self.information_loss > IL_max:
            return -float("inf")
        return self.information_gain

    def solved_by_time(self, t):
        if t < 0 or self.steps == 0 or (len(self.assignments) == 0 and self.steps > 0):
            return 0

        # if len(self.assignments) == 0 and self.steps > 0:
        #     return np.inf


        initial_entropy = np.sum(
            [variable.get_initial_entropy() for variable in self.variables]
        )
        initial_size = 2**initial_entropy


        number_culled, lost_assignments = get_sizes_from_culling(
            self.assignments, self.information_loss
        )
        p_culled = (number_culled + lost_assignments) / initial_size
        rate_of_culling = p_culled / self.steps
        # print()
        # print(f"steps = {self.steps}", f"number_culled = {number_culled:.2f}", f"lost_assignments = {lost_assignments:.2f}")
        # print(f"initial_entropy = {initial_entropy:.2f}, initial_size = {initial_size:.2f}, current_size={len(self.assignments)}, rate_of_culling = {rate_of_culling:.3f}")

        solved_by_t = expected_deductions_by_time(self.assignments, rate_of_culling, t) if rate_of_culling > 0 else 0
        return solved_by_t

    def expected_discounted_marks(self, gamma, T=1):
        expected_marks = self.solved_by_time(0)
        discounted_marks = 0
        for t in range(1, T + 1):
            expected_new_marks = self.solved_by_time(t) - expected_marks
            expected_marks += expected_new_marks
            discounted_marks += expected_new_marks * gamma**t
        return discounted_marks

    def remove_solved_variables(self):
        n_vars_before = len(self.assignments[0]) if self.assignments else 0
        per_var_IL = self.information_loss / n_vars_before if n_vars_before > 0 else 0
        self.assignments = remove_solved_variables(self.assignments)
        n_vars_after = len(self.assignments[0]) if self.assignments else 0
        self.information_loss -= per_var_IL * (n_vars_after - n_vars_before)


class Agent:
    def __init__(
        self,
        constraints,
        memory_capacity=np.inf,
        R_init=1.0,
        ILtol_init=0.0,
        gamma=1.0,
        max_steps=250,
        reset_variables = True,
        enforce_locality = True,
        **kwargs,
    ):
        self.constraints = set(constraints)
        self.variables = set().union(*[c.variables for c in self.constraints])
        if reset_variables:
            self.reset_all_variables()


        self.initial_variables = VariableState(self.variables)
        self.solved_variables = set([v for v in self.variables if v.is_assigned()])

        self.memory_capacity, self.R_init, self.ILtol_init, self.gamma = (
            memory_capacity,
            R_init,
            ILtol_init,
            gamma        )

        self.R_current = R_init
        self.ILtol_current = ILtol_init
        self.max_steps = max_steps
        self.max_IL = np.sum([v.get_initial_entropy() for v in self.variables])

        self.current_subproblem = SubProblem()
        self.previous_subproblems = []
        self.total_steps = 0
        self.information_gain_total = 0
        self.information_loss_total = 0
        self.first_step_EDRs = []
        self.enforce_locality = enforce_locality
        self.last_proposal = None
        self.last_proposal_objects = None  # (constraint_obj, set[Variable])

        self.variables_of_interest = kwargs.get('variables_of_interest', None)
        self.allow_marking = kwargs.get('allow_marking', True)
        self.subproblems_without_progress = 0

    def reset(self):
        self.previous_subproblems.append(self.current_subproblem.copy())
        self.current_subproblem = SubProblem()

    def reset_all_variables(self):
        for v in self.variables:
            v.unassign()

    def accept(self, new_subproblem):
        self.current_subproblem = new_subproblem.copy()

    def accept_or_reject(self, new_subproblem):

        if new_subproblem.V(self.ILtol_current) > self.current_subproblem.V(
            self.ILtol_current
        ):
            self.accept(new_subproblem)
            return Result.ACCEPT
        else:
            return Result.REJECT

    def get_active_constraints(self):
        return set([c for c in self.constraints if c.is_active()])

    def sample_constraint(self):
        if self.variables_of_interest:
            constraints = self.get_active_constraints()
            constraints = set([c for c in constraints if len(set(c.variables) & self.variables_of_interest) > 0])
            return random.choice(list(constraints))
        else:
            return random.choice(list(self.get_active_constraints()))

    def sample_related_constraint(self):


        related_constraints = set(
            [
                c
                for c in self.get_active_constraints()
                if ((len(set(c.variables) & self.current_subproblem.variables) > 0) and\
                      (c not in self.current_subproblem.unsimplified_constraints))
            ]
        )

        for c in related_constraints:
            if c in self.current_subproblem.unsimplified_constraints:
                raise ValueError(f"Constraint {c} is in the unsimplified constraints but not in the related constraints")
        # if not related_constraints:
        #     return self.sample_constraint()
        if not related_constraints:
            return None
        return random.choice(list(related_constraints))


    def sample_new_constraint(self):
        if (not self.enforce_locality) or (not self.current_subproblem.variables):
            return self.sample_constraint()
        else:
            return self.sample_related_constraint()


    def sample_variables(self, constraint, simplify=True):
        if not simplify:
            return constraint.get_unassigned()

        possible_new_vars = constraint.get_unassigned() - self.current_subproblem.variables



        min_size = 0 if self.current_subproblem.variables else 1


        num_vars = (
            random.randint(min_size, len(possible_new_vars))
            if min_size < len(possible_new_vars)
            else min_size
        )

        possible_new_vars_of_interest = self.variables_of_interest & set(possible_new_vars) if self.variables_of_interest else set()
        if possible_new_vars_of_interest and num_vars > 0:
            new_vars_of_interest = set(random.sample(list(possible_new_vars_of_interest), 1))
            possible_new_vars = possible_new_vars - set(new_vars_of_interest)
            other_vars = set(random.sample(list(possible_new_vars), num_vars - 1))
            new_vars = new_vars_of_interest | other_vars
        else:
            new_vars = set(random.sample(list(possible_new_vars), num_vars))

        return new_vars

    def propose_new_subproblem(self, constraint, variables):
        new_subproblem = self.current_subproblem.copy().add(constraint, variables, memory_capacity=self.memory_capacity)
        if new_subproblem.steps <= 1:
            EDR = new_subproblem.expected_discounted_marks(self.gamma)
            if not np.isnan(EDR):
                self.first_step_EDRs.append(EDR)
        return new_subproblem


    def propose_step(self, simplify=True):

        new_constraint = self.sample_new_constraint()

        if not new_constraint:
            self.last_proposal = (None, None, None, False)
            self.last_proposal_objects = (None, None)
            return Result.NO_MORE_CONSTRAINTS

        new_variables = self.sample_variables(new_constraint, simplify=simplify)

        if new_constraint.test_contradiction():

            # Handle contradiction and record proposal
            self.last_proposal = (str(new_constraint), [str(v) for v in new_variables], None, True)
            self.last_proposal_objects = (new_constraint, set(new_variables))
            self.handle_contradiction(new_constraint)

            return Result.CONTRADICTION

        new_subproblem = self.propose_new_subproblem(new_constraint, new_variables)
        result = self.accept_or_reject(new_subproblem)
        self.last_proposal = (str(new_constraint), [str(v) for v in new_variables], result.name, False)
        self.last_proposal_objects = (new_constraint, set(new_variables))
        return result

    def increment_steps(self):
        self.total_steps += 1
        self.current_subproblem.increment_step()

    def mark_solved_variables(self):
        if self.current_subproblem.found_contradiction:
            return {}

        solved_variables = get_solved_variables(self.current_subproblem.assignments)
        n_vars = len(self.current_subproblem.assignments[0]) if self.current_subproblem.assignments else 0
        per_var_IL = self.current_subproblem.information_loss / n_vars if n_vars > 0 else 0

        if (not self.allow_marking) and (self.variables_of_interest and len(self.variables_of_interest) > 0):
            solved_variables = {v: solved_variables[v] for v in solved_variables if v in self.variables_of_interest}

        if not solved_variables:
            self.subproblems_without_progress += 1
            return {}
        else:
            self.subproblems_without_progress = 0

        for v in solved_variables:
            v.assign(solved_variables[v])
            self.solved_variables.add(v)
            self.information_loss_total += per_var_IL
            self.information_gain_total += v.get_initial_entropy()

        self.current_subproblem.remove_solved_variables()
        return solved_variables

    def randomly_solve(self, variables=None):
        if variables is None:
            variables = [v for v in self.variables if not v.is_assigned()]

        solved_variables = {}
        for v in variables:
            value = random.choice(list(v.domain))
            self.information_loss_total += v.get_initial_entropy()
            self.information_gain_total += v.get_initial_entropy()
            v.assign(value)
            self.solved_variables.add(v)
            solved_variables[v] = value

        return solved_variables


    def handle_contradiction(self, constraint):
        unassigned = constraint.fix_contradiction()
        for var in unassigned:
            if var not in self.solved_variables:
                raise ValueError(f"Variable {var} is not solved!")
            self.solved_variables.remove(var)
        self.current_subproblem.remove_solved_variables()
        self.current_subproblem.found_contradiction = True
        return unassigned


    def check_if_finished(self):
        if self.total_steps >= self.max_steps:
            return True
        if len(self.solved_variables) == len(self.variables):
            return True
        if self.variables_of_interest and len(self.variables_of_interest) > 0:
            if all(v.is_assigned() for v in self.variables_of_interest):
                return True
        return False

    def continue_with_subproblem(self, result):
        if self.check_if_finished():
            return False
        if result == Result.CONTRADICTION:
            return False
        if result == Result.NO_MORE_CONSTRAINTS:
            return False

        return True





def agent_loop(
    constraints,
    true_assignments,
    memory_capacity=np.inf,
    R_init=1,
    ILtol_init=0,
    gamma=1,
    max_steps=100,
    save_heatmaps=False,
    print_output=False,
    enforce_locality=True,
    **kwargs
):
    agent = Agent(
        constraints,
        memory_capacity=memory_capacity,
        R_init=R_init,
        ILtol_init=ILtol_init,
        gamma=gamma,
        max_steps=max_steps,
        enforce_locality=enforce_locality,
        **kwargs
    )



    subproblem_steps, constraint_accepted = [], []
    n_errors = 0
    while not agent.check_if_finished():
        agent.reset()

        print("=" * 80)
        print("NEW SUBPROBLEM")
        print(f"steps = {agent.total_steps}, subproblems_without_progress = {agent.subproblems_without_progress}")
        print("=" * 80)
        print() 
        
        subproblem_steps = subproblem_steps + [0]
        while True:
            print("")
            print("~"*25)

            expected_discounted_marks = (
                agent.current_subproblem.expected_discounted_marks(gamma)
            )
    
            print(f"total steps = {agent.total_steps}, subproblem steps = {agent.current_subproblem.steps}")
            print(f"R = {agent.R_current:.2f}, ILtol = {agent.ILtol_current:.2f}")
            
            print(f"solved_variables = {agent.solved_variables}")
            print(f"IL: {agent.current_subproblem.information_loss:.2f}, IG: {agent.current_subproblem.information_gain:.2f}, EDM = {expected_discounted_marks:.4f}")
            print(f"constraints = {agent.current_subproblem.constraints}")
            print_assignments(agent.current_subproblem.assignments)


            if expected_discounted_marks < agent.R_current and agent.current_subproblem.steps > 0:
                print(f"breaking because EDM < R", f"EDM = {expected_discounted_marks:.4f}, R = {agent.R_current:.2f}")
                break

            agent.increment_steps()
            subproblem_steps[-1] += 1
            result = agent.propose_step(simplify=True)
            accept = 1 if result == Result.ACCEPT else 0
            constraint_accepted.append(accept)

            prop = agent.last_proposal if agent.last_proposal else (None, None, None, False)
            print(f"proposal = constraint:{prop[0]}, vars:{prop[1]}, decision:{prop[2]}, contradiction:{prop[3]} | accepted={bool(accept)}")


            if not agent.continue_with_subproblem(result):
                print("BREAKING: ", result.name)
                break


        agent.mark_solved_variables()



    return agent




if __name__ == "__main__":


    constraints, true_assignments = generate_random_constraints(n_variables=12, n_constraints=10,
     p_inequality=0.1, avg_size=3, sd_size=2)



    variables = sorted(list(set().union(*[c.variables for c in constraints])), key=lambda x: x.name)
    variables_of_interest = None

    agent = agent_loop(constraints, true_assignments, memory_capacity=25, R_init=0.1,
                ILtol_init=1,  max_steps=200,  print_output=True, enforce_locality=True, gamma=1,
                variables_of_interest=variables_of_interest, allow_marking=False )

    solved_variables = agent.solved_variables
    for v in true_assignments:
        if v in solved_variables:
            print(v.name, true_assignments[v], v.value, 1*(true_assignments[v] == v.value))
        else:
            print(v.name, true_assignments[v], "?", "")

    # print(variables_of_interest)

    # memory_capacities = [4, 8, 16, 32]
    # R_inits = [0.1, 0.5, 2]
    # delta_Rs = [0.0]
    # ILtol_inits = [0.1, 0.5, 4]
    # gammas = [1]
    # max_steps = 200
    # n_simulations = 15

    # savefile = Path(__file__).parent / "simulations" / "sam_and_tony_week_2.csv"
    # alldata = run_param_sweeps(memory_capacities, R_inits, delta_Rs, ILtol_inits, gammas, max_steps, n_simulations, savefile=savefile  )
