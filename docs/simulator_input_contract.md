\# Adaptive Multi-Core CPU Scheduler

\# Simulator Input Contract



\## 1. Purpose



This document freezes the data representation and information boundaries that the event-driven simulator will use.



The simulator must use only validated information from the four prepared datasets and the finalized workload scenario manifest.



The raw datasets remain immutable.



The simulator must not expose future information to the scheduling policy.



\## 2. Simulator Design



The simulator is event-driven.



The primary resources are:



\- CPU

\- Memory



The initial simulator is non-preemptive.



The simulator maintains:



\- waiting tasks

\- running tasks

\- machine CPU availability

\- machine memory availability

\- scheduling events

\- task completion events

\- task resource release

\- scheduling metrics



The scheduler operates whenever a meaningful scheduling event occurs rather than advancing through unnecessary fixed time ticks.



\## 3. Scenario Selection



A simulator episode is associated with one scenario from:



dataset/workload\_analysis/workload\_scenario\_manifest\_final.csv



Each scenario contains:



\- scenario\_id

\- split

\- scenario\_type

\- scenario\_axis

\- regime

\- time\_bin

\- time\_start

\- time\_end

\- workload\_intensity

\- resource\_dominance

\- variability\_score

\- intensity\_regime

\- resource\_regime

\- variability\_regime

\- selection\_seed



The scenario manifest is deterministic.



The scenario selection seed is fixed.



\## 4. Task Selection



A scenario window is defined by:



time\_start

time\_end



Task instances are selected from the corresponding batch\_instance split according to their start\_time.



The initial task inclusion rule is:



start\_time >= time\_start

and

start\_time < time\_end



The upper boundary is exclusive.



This prevents a task from being assigned simultaneously to adjacent scenario windows.



Tasks are ordered chronologically by start\_time.



When multiple tasks have the same start\_time, ordering must use deterministic secondary keys.



The simulator must not randomly reorder tasks unless randomization is explicitly part of a controlled experiment.



\## 5. Task Representation



The validated batch\_instance fields are:



\- instance\_name

\- task\_name

\- job\_name

\- task\_type

\- status

\- start\_time

\- end\_time

\- machine\_id

\- seq\_no

\- total\_seq\_no

\- cpu\_avg

\- cpu\_max

\- mem\_avg

\- mem\_max

\- duration



The simulator task representation must distinguish between:



\### Scheduling inputs



\- task\_type

\- cpu requirement

\- memory requirement

\- duration or runtime estimate

\- arrival/start time

\- task identity required internally for event tracking



\### Ground-truth or evaluation information



\- original machine\_id

\- original end\_time

\- observed duration

\- completion information



\### Metadata



\- task\_name

\- job\_name

\- instance\_name

\- sequence fields



The scheduling policy must not receive the original machine\_id as a machine-selection feature.



The policy should select machines from machine characteristics and current machine state.



\## 6. CPU Requirement



cpu\_avg is retained as a task CPU demand representation.



cpu\_max is retained as an additional task resource characteristic.



The original values are not clipped or modified.



Resource normalization belongs to the later observation construction stage.



The simulator must preserve the distinction between:



\- raw resource requirement

\- normalized policy representation



\## 7. Memory Requirement



mem\_avg is retained as a task memory demand representation.



mem\_max is retained as an additional task resource characteristic.



The original values are not clipped or modified.



Resource normalization belongs to the later observation construction stage.



\## 8. Missing Resource Values



Missing resource values remain missing in the prepared data.



The simulator must not silently replace missing values with arbitrary constants.



A simulator input policy for missing task resource values must be explicitly defined before production experiments.



The handling rule must be deterministic.



\## 9. Runtime Representation



The project uses a hybrid observed/estimated runtime design.



The simulator may use:



\- observed duration when valid

\- estimated runtime when an estimate is required



Resource-dependent slowdown may be applied only where supported by trace evidence.



The simulator must not use future completion information when making a scheduling decision.



\## 10. Machine Representation



Machines are represented using validated machine characteristics rather than exposing raw machine IDs to the policy.



Machine information is derived from:



machine\_meta



and, where appropriate for current-state information:



machine\_usage



Relevant machine characteristics include:



\- CPU capacity

\- memory capacity

\- failure-domain information where applicable

\- current CPU utilization

\- current memory utilization

\- current resource availability



Raw machine identifiers may be retained internally for bookkeeping and event processing.



Raw machine identifiers must not be treated as learned semantic features.



\## 11. Temporal Machine Information



Machine state available to the scheduler must correspond to information available at the current simulation time.



Future machine\_usage observations must not be exposed.



Historical machine information may be used only when it is available before the decision time and when the feature construction is explicitly defined.



\## 12. Feasible Machine Filtering



For each scheduling decision:



1\. Determine machines capable of satisfying the task requirements.

2\. Remove machines that cannot satisfy current CPU requirements.

3\. Remove machines that cannot satisfy current memory requirements.

4\. Apply any other validated feasibility constraints.

5\. Rank or select the feasible candidates deterministically.

6\. Retain at most K = 20 candidates.



K is fixed at 20.



If fewer than 20 machines are feasible, only the available feasible machines are retained.



If no machine is feasible, the simulator must invoke explicit invalid-action or deferred-scheduling handling.



\## 13. Waiting Queue



The simulator maintains an adaptive waiting-task queue.



Tasks enter the queue at their arrival time.



Completed tasks leave the running set and release resources.



The PPO policy will eventually select a task from the currently waiting tasks.



The queue must support deterministic inspection and selection.



The initial simulator must not assume a fixed FIFO scheduling policy because the RL scheduler is intended to learn task selection.



\## 14. Scheduling Action



The final policy action is sequential:



1\. Select task.

2\. Select machine.

3\. Select continuous resource-allocation factor.



The environment must validate every action.



Invalid actions must not silently modify the simulation state.



\## 15. Resource Allocation



The continuous resource-allocation factor controls the allocation decision subject to resource feasibility.



The allocation must remain within valid CPU and memory limits.



An allocation that exceeds available machine resources must be rejected or penalized according to the finalized action-handling rule.



\## 16. Event Types



The initial event system must support:



\### Task arrival



A task becomes available to the scheduler.



\### Task completion



A running task completes execution.



\### Resource release



CPU and memory resources occupied by a completed task become available.



Additional event types may be introduced only when justified by the simulator design.



\## 17. Non-Preemption



Preemption is disabled in the initial simulator.



Once a task begins execution:



\- its assigned machine remains fixed

\- its allocation remains governed by the execution model

\- it is not removed from the machine by a later scheduling decision



Resources are released when the task completes.



\## 18. Simulator State



The internal simulator state must contain sufficient information to advance the event-driven system.



At minimum:



\- current simulation time

\- waiting tasks

\- running tasks

\- machine resource availability

\- machine utilization state

\- next task-arrival information

\- next task-completion information

\- task execution state

\- accumulated performance metrics

\- deterministic random state



\## 19. Policy Observation Boundary



The simulator state is larger than the policy observation.



The policy may receive only information available at the current decision time.



The policy must not receive:



\- future task arrivals

\- future task completions

\- future machine utilization

\- future machine failures

\- future workload statistics

\- original machine assignment when it would leak the historical scheduling decision



The observation representation will be finalized in the structured-observation stage.



\## 20. Evaluation Information



The simulator may internally retain information required to calculate:



\- waiting time

\- execution time

\- turnaround time

\- throughput

\- makespan

\- CPU utilization

\- memory utilization

\- resource imbalance

\- overload

\- starvation

\- fairness

\- invalid actions



These quantities may be used for reward calculation and evaluation.



They must not automatically become policy observations.



\## 21. Workload Regimes



The finalized workload regime dimensions are:



\- Low

\- Medium

\- High

\- Highly Variable

\- CPU-heavy

\- Memory-heavy



Regimes are allowed to overlap.



A scenario may therefore belong to more than one regime dimension.



The regime definitions are learned from training workload statistics and subsequently applied to validation and testing.



\## 22. Scenario Types



Two scenario families are supported:



\### Natural-frequency



Preserve the observed workload-window distribution.



\### Balanced



Select controlled comparable numbers of eligible workload windows for a particular regime dimension.



The final scenario manifest is:



dataset/workload\_analysis/workload\_scenario\_manifest\_final.csv



\## 23. Dataset Split Usage



Training scenarios use the training split.



Validation scenarios use the validation split.



Testing scenarios use the testing split.



No task instance may be copied from one split into another.



The chronological split must remain intact.



Test data must not be used to tune simulator or PPO design decisions.



\## 24. Deterministic Seeding



The simulator must support deterministic seeds.



The same:



\- scenario

\- initial state

\- random seed

\- action sequence



must produce reproducible simulation results.



Randomness must be controlled explicitly.



\## 25. Simulator Metrics



The simulator must expose enough information to calculate:



\- task waiting time

\- task turnaround time

\- execution time

\- throughput

\- makespan

\- CPU utilization

\- memory utilization

\- resource imbalance

\- overload

\- starvation

\- fairness

\- invalid-action frequency



Jain fairness and waiting-time distribution statistics are part of the project evaluation.



\## 26. Classical Baseline Compatibility



The simulator must be scheduler-agnostic.



PPO and classical baselines must use the same simulator.



The baseline schedulers must receive equivalent observable information.



The planned baselines include:



\- Least Loaded

\- Best Fit

\- Resource-aware Best Fit



Additional baselines may include:



\- Random

\- Round Robin

\- First Fit



\## 27. RL Integration Boundary



The simulator must not contain PPO-specific model logic.



The environment is responsible for:



\- state transition

\- event processing

\- feasibility

\- action validation

\- resource accounting

\- execution

\- metric collection

\- reward inputs



The PPO policy is responsible for:



\- observation processing

\- task selection

\- machine selection

\- resource allocation decision



\## 28. Required Simulator Tests



Before connecting PPO, test:



\- task arrival

\- task completion

\- resource allocation

\- resource release

\- queue transitions

\- machine feasibility

\- K = 20 filtering

\- invalid actions

\- zero-duration tasks

\- missing-resource handling

\- simultaneous events

\- deterministic seeding

\- no-resource-feasible conditions

\- chronological event processing



The simulator must pass these tests before PPO implementation.



\## 29. Information Leakage Rules



The simulator must never use future information to make a current scheduling decision.



In particular:



\- future task arrivals are unavailable

\- future machine usage is unavailable

\- future task completion is unavailable

\- future workload regime labels are unavailable

\- test-period statistics are unavailable during training decisions



Any historical feature must be constructed only from information available before the decision time.



\## 30. Frozen Design Constraints



The following project decisions remain fixed:



\- PPO is the principal RL algorithm.

\- The environment is event-driven.

\- CPU and memory are the primary resources.

\- The queue is adaptive.

\- The action is task → machine → continuous resource allocation.

\- K = 20 is the maximum candidate count.

\- Task-to-machine cross-attention will be used later.

\- LSTM temporal memory will be used later.

\- The initial simulator is non-preemptive.

\- Reward will be normalized and multi-objective.

\- Classical baselines use the same simulator.

\- Multiple fixed seeds will be used.

\- Statistical reporting is required.

\- No ablation study is included in the current scope.



\## 31. Pre-Implementation Rule



This contract must be treated as the simulator implementation boundary.



Any change to:



\- task representation

\- machine representation

\- temporal information

\- action semantics

\- resource semantics

\- scenario selection

\- leakage rules



must be explicitly documented before implementation.



\## 32. Current Input Artifacts



Validated inputs:



dataset/preprocessed/batch\_task.csv



dataset/preprocessed/machine\_meta.csv



dataset/cleaned/machine\_usage.csv



dataset/preprocessed/batch\_instance\_training.csv



dataset/preprocessed/batch\_instance\_validation.csv



dataset/preprocessed/batch\_instance\_testing.csv



dataset/data\_dictionary.csv



dataset/workload\_analysis/workload\_regime\_profiles.csv



dataset/workload\_analysis/workload\_regime\_thresholds.csv



dataset/workload\_analysis/workload\_scenario\_manifest\_final.csv



dataset/workload\_analysis/workload\_scenario\_manifest\_validation.csv

