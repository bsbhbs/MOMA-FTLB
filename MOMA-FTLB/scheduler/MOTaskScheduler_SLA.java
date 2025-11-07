package scheduler;

import simulation.MOSimulation_SLA;

public abstract class MOTaskScheduler_SLA {
    protected MOSimulation_SLA sim;

    public MOTaskScheduler_SLA(MOSimulation_SLA sim) {
        this.sim = sim;
    }

    public abstract int[] schedule(int MAX_FES);
}
