package scheduler;



import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import simulation.MOSimulation_SLA;
import simulation.Util;

public class MOMA_Scheduler_SLA extends MOTaskScheduler_SLA {//Four targets plus an SLA version

    public MOMA_Scheduler_SLA(MOSimulation_SLA sim) {
        super(sim);
        rng = sim.getRng();
    }

    @Override
    public int[] schedule(int MAX_FES) {
        max_evaluations = MAX_FES;
        check_parameters();
        init();
        run_wo();
        int[] mapping = Util.discretizeSol(bestWalrus.getPosition());
        return mapping;
    }

    public void init() {
        eval = 0;
        population = new ArrayList<>(populationSize);
        archive = new ArrayList<>(); // Initialize the archive
        for (int i = 0; i < populationSize && eval < max_evaluations; i++) {
            Walrus walrus = new Walrus(dim);
            walrus.init(rng, min_bound, max_bound);
            walrus.cost = sim.predictCostValue(Util.discretizeSol(walrus.getPosition()));
            eval++;
            population.add(walrus);
            // Add each walrus to the archive during initialization
            updateArchive(walrus);
        }
        updateBestSolutions();
        if (bestWalrus == null) {
            bestWalrus = population.get(0);
        }
        if (secondBestWalrus == null && population.size() > 1) {
            secondBestWalrus = population.get(1);
        }
    }

    private boolean check_parameters() {
        dim = sim.getNumOfCloudlets();
        populationSize = 50; //Population size
        min_bound = 0;
        max_bound = sim.getNumOfVMs() - 1;
        eval = 0;
        return true;
    }

    public void run_wo() {
        while (eval < max_evaluations) {
            double alpha_val = 1.0 - (eval / (double) max_evaluations);

            // Safety inspection
            if (bestWalrus == null || secondBestWalrus == null) {
                updateBestSolutions();
                if (bestWalrus == null) break; // If it is still null, terminate the loop
            }

            for (int i = 0; i < populationSize && eval < max_evaluations; i++) {
                Walrus current = population.get(i);

                // Calculate danger signals
                double A = 2 * alpha_val;
                double R = 2 * rng.nextDouble() - 1;
                double dangerSignal = A * R;

                if (Math.abs(dangerSignal) >= 1) {
                    // Migration behavior
                    int m = rng.nextInt(populationSize);
                    int n = rng.nextInt(populationSize);
                    while (m == i) m = rng.nextInt(populationSize);
                    while (n == i || n == m) n = rng.nextInt(populationSize);

                    Walrus Xm = population.get(m);
                    Walrus Xn = population.get(n);

                    double beta = 1.0 - 1.0 / (1 + Math.exp(-10 * (0.5 - alpha_val)));
                    double r3 = rng.nextDouble();
                    double stepFactor = beta * r3 * r3;

                    double[] newPosition = new double[dim];
                    for (int d = 0; d < dim; d++) {
                        newPosition[d] = current.position[d] +
                                (Xm.position[d] - Xn.position[d]) * stepFactor;
                        newPosition[d] = Math.max(min_bound, Math.min(max_bound, newPosition[d]));
                    }

                    double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                    eval++;

                    if (paretoDominance(newCost, current.cost)) {
                        current.position = newPosition;
                        current.cost = newCost;
                        updateArchive(current);
                    }
                } else {
                    // Development stage
                    double safetySignal = rng.nextDouble();

                    if (safetySignal >= 0.5) {
                        // Performative behavior
                        int adultCount = (int) (populationSize * 0.9);
                        int maleCount = adultCount / 2;
                        int femaleCount = adultCount - maleCount;

                        if (i < maleCount) {
                            // Male: Halton sequence
                            double[] newPosition = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                newPosition[d] = min_bound + (max_bound - min_bound) * halton(i, prime(d));
                            }
                            double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                            eval++;
                            if (paretoDominance(newCost, current.cost)) {
                                current.position = newPosition;
                                current.cost = newCost;
                                updateArchive(current);
                            }
                        } else if (i < maleCount + femaleCount) {
                            // Female
                            int maleIndex = rng.nextInt(maleCount);
                            Walrus male = population.get(maleIndex);
                            double a = rng.nextDouble();

                            double[] newPosition = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                // Add null check
                                double bestPos = bestWalrus != null ? bestWalrus.position[d] : current.position[d];
                                newPosition[d] = current.position[d] +
                                        a * (male.position[d] - current.position[d]) +
                                        (1 - a) * (bestPos - current.position[d]);
                                newPosition[d] = Math.max(min_bound, Math.min(max_bound, newPosition[d]));
                            }

                            double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                            eval++;
                            if (paretoDominance(newCost, current.cost)) {
                                current.position = newPosition;
                                current.cost = newCost;
                                updateArchive(current);
                            }
                        } else {
                            // Childhood
                            double P = rng.nextDouble();
                            double[] O = new double[dim];
                            double[] levy = levyFlight();
                            for (int d = 0; d < dim; d++) {
                                // Add null check
                                double bestPos = bestWalrus != null ? bestWalrus.position[d] : current.position[d];
                                O[d] = bestPos + current.position[d] * levy[d];
                                O[d] = Math.max(min_bound, Math.min(max_bound, O[d]));
                            }

                            double[] newPosition = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                newPosition[d] = (O[d] - current.position[d]) * P;
                                newPosition[d] = Math.max(min_bound, Math.min(max_bound, newPosition[d]));
                            }

                            double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                            eval++;
                            if (paretoDominance(newCost, current.cost)) {
                                current.position = newPosition;
                                current.cost = newCost;
                                updateArchive(current);
                            }
                        }
                    } else {
                        // Foraging behavior
                        if (Math.abs(dangerSignal) >= 0.5) {
                            // Aggregation behavior
                            double beta_val = 1.0 - 1.0 / (1 + Math.exp(-10 * (0.5 - alpha_val)));
                            double r5 = rng.nextDouble();
                            double a_val = beta_val * r5 - beta_val;
                            double theta = rng.nextDouble() * Math.PI;
                            double b_val = Math.tan(theta);

                            // Calculate X1 - Add null check
                            double[] X1 = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                double bestPos = bestWalrus != null ? bestWalrus.position[d] : current.position[d];
                                double diff = Math.abs(bestPos - current.position[d]);
                                X1[d] = bestPos - a_val * b_val * diff;
                                X1[d] = Math.max(min_bound, Math.min(max_bound, X1[d]));
                            }

                            // Calculate X2 - Add null check
                            double a2_val = beta_val * rng.nextDouble() - beta_val;
                            double b2_val = Math.tan(rng.nextDouble() * Math.PI);
                            double[] X2 = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                double secondBestPos = secondBestWalrus != null ? secondBestWalrus.position[d] : current.position[d];
                                double diff = Math.abs(secondBestPos - current.position[d]);
                                X2[d] = secondBestPos - a2_val * b2_val * diff;
                                X2[d] = Math.max(min_bound, Math.min(max_bound, X2[d]));
                            }

                            // The new position = (X1 + X2)/2
                            double[] newPosition = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                newPosition[d] = (X1[d] + X2[d]) / 2.0;
                                newPosition[d] = Math.max(min_bound, Math.min(max_bound, newPosition[d]));
                            }

                            double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                            eval++;
                            if (paretoDominance(newCost, current.cost)) {
                                current.position = newPosition;
                                current.cost = newCost;
                                updateArchive(current);
                            }
                        } else {
                            // Escape behavior - Add null check
                            double R_val = 2 * rng.nextDouble() - 1;
                            double r4 = rng.nextDouble();

                            double[] newPosition = new double[dim];
                            for (int d = 0; d < dim; d++) {
                                double bestPos = bestWalrus != null ? bestWalrus.position[d] : current.position[d];
                                double diff = Math.abs(bestPos - current.position[d]);
                                newPosition[d] = current.position[d] * R_val - diff * (r4 * r4);
                                newPosition[d] = Math.max(min_bound, Math.min(max_bound, newPosition[d]));
                            }

                            double[] newCost = sim.predictCostValue(Util.discretizeSol(newPosition));
                            eval++;
                            if (paretoDominance(newCost, current.cost)) {
                                current.position = newPosition;
                                current.cost = newCost;
                                updateArchive(current);
                            }
                        }
                    }
                }
            }
        }

        bestWalrus = findBestInArchive();
    }

    private boolean paretoDominance(double[] solution1, double[] solution2) {
        boolean betterOrEqual = true;
        boolean strictlyBetter = false;


        // Objective 1: Execution time (the shorter, the better)
        if (solution1[0] > solution2[0]) {
            betterOrEqual = false;
        } else if (solution1[0] < solution2[0]) {
            strictlyBetter = true;
        }

        // Objective 2: Resource utilization rate (The higher, the better)
        if (solution1[1] < solution2[1]) {
            betterOrEqual = false;
        } else if (solution1[1] > solution2[1]) {
            strictlyBetter = true;
        }

        // Objective 3: Execution Cost (the lower, the better)
        if (solution1[2] > solution2[2]) {
            betterOrEqual = false;
        } else if (solution1[2] < solution2[2]) {
            strictlyBetter = true;
        }

        // Objective 4: Task Success Rate (The higher, the better)
        if (solution1[3] < solution2[3]) {
            betterOrEqual = false;
        } else if (solution1[3] > solution2[3]) {
            strictlyBetter = true;
        }

        return betterOrEqual && strictlyBetter;
    }

    private void updateArchive(Walrus newWalrus) {
        // Check whether it is dominated by the solutions in the archive
        boolean dominated = false;
        for (Walrus w : archive) {
            if (paretoDominance(w.cost, newWalrus.cost)) {
                dominated = true;
                break;
            }
        }

        if (!dominated) {
            // Add a new solution and remove the solutions it governs
            archive.add(newWalrus);
            List<Walrus> toRemove = new ArrayList<>();
            for (Walrus w : archive) {
                if (paretoDominance(newWalrus.cost, w.cost)) {
                    toRemove.add(w);
                }
            }
            archive.removeAll(toRemove);
        }

        // If the archive is too large, crop it according to the congestion distance
        if (archive.size() > maxArchiveSize) {
            calculateCrowdingDistance();
            archive.sort((w1, w2) -> Double.compare(w2.crowdingDistance, w1.crowdingDistance));
            archive = archive.subList(0, maxArchiveSize);
        }
    }

    private void calculateCrowdingDistance() {
        int size = archive.size();
        if (size == 0) return;

        // Reset the crowded distance
        for (Walrus w : archive) {
            w.crowdingDistance = 0;
        }

        // Calculate for each target
        for (int objIndex = 0; objIndex < 4; objIndex++) { // There are four goals now
            final int idx = objIndex;
            archive.sort((w1, w2) -> Double.compare(w1.cost[idx], w2.cost[idx]));

            // The boundary solution has an infinite distance
            archive.get(0).crowdingDistance = Double.POSITIVE_INFINITY;
            archive.get(size - 1).crowdingDistance = Double.POSITIVE_INFINITY;

            // Calculate the crowding distance of the intermediate solution
            double minObj = archive.get(0).cost[idx];
            double maxObj = archive.get(size - 1).cost[idx];
            double range = maxObj - minObj;
            if (range == 0) range = 1; // Avoid dividing by 0

            for (int i = 1; i < size - 1; i++) {
                double previousObjValue = archive.get(i - 1).cost[idx];
                double nextObjValue = archive.get(i + 1).cost[idx];
                archive.get(i).crowdingDistance += (nextObjValue - previousObjValue) / range;
            }
        }
    }

    private Walrus findBestInArchive() {
        if (archive.isEmpty()) return population.get(0);

        calculateCrowdingDistance();
        Walrus best = archive.get(0);
        for (Walrus w : archive) {
            if (w.crowdingDistance > best.crowdingDistance) {
                best = w;
            }
        }
        return best;
    }

    private void updateBestSolutions() {
        if (archive.size() < 2) return;

        calculateCrowdingDistance();
        archive.sort((w1, w2) -> Double.compare(w2.crowdingDistance, w1.crowdingDistance));

        if (archive.size() > 0) bestWalrus = archive.get(0);
        if (archive.size() > 1) secondBestWalrus = archive.get(1);
    }

    // Halton sequence generator
    private double halton(int index, int base) {
        double f = 1.0;
        double r = 0.0;

        while (index > 0) {
            f /= base;
            r += f * (index % base);
            index = index / base;
        }

        return r;
    }

    private int prime(int index) {
        // Return the index prime number
        int[] primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
        return primes[index % primes.length];
    }

    // Levy Flight Generator
    private double[] levyFlight() {
        double[] step = new double[dim];
        double sigma_x = Math.pow((gamma(1.5) * Math.sin(Math.PI * 0.75)) /
                (gamma(1.0) * 1.5 * Math.pow(2, 0.75)), 1/1.5);

        for (int i = 0; i < dim; i++) {
            double u = rng.nextGaussian() * sigma_x;
            double v = rng.nextGaussian();
            step[i] = 0.01 * (u / Math.pow(Math.abs(v), 1/1.5));
        }
        return step;
    }

    private double gamma(double x) {
        // Gamma function
        return Math.sqrt(2 * Math.PI / x) * Math.pow((x / Math.E), x);
    }

    // Walrus individuals
    class Walrus {
        double[] position;
        double[] cost;
        double crowdingDistance;

        Walrus(int dim) {
            position = new double[dim];
            cost = new double[4]; // There are four goals now
            crowdingDistance = 0;
        }

        void init(Random rng, double min, double max) {
            for (int i = 0; i < position.length; i++) {
                position[i] = min + (max - min) * rng.nextDouble();
            }
        }

        double[] getPosition() {
            return position;
        }
    }

    // Parameter
    private int populationSize = 50;
    private int dim;
    private int max_evaluations;
    private int eval;
    private Random rng;
    private double min_bound;
    private double max_bound;

    private List<Walrus> population;
    private List<Walrus> archive = new ArrayList<>();
    private final int maxArchiveSize = 300;

    private Walrus bestWalrus;
    private Walrus secondBestWalrus;
}
