# SchedulNet: Neural Networks vs. Classic Scheduling Approaches

Honestly, I have no clue what I'm trying to do here. I'm just benchmarking traditional OS algorithms against neural networks and AI agents because I don't know what I'm doing. I'll keep updating this repo whenever I find something interesting in this space.

## What's This All About?

This is an experimental project where I'm trying to see if AI/ML approaches can outperform traditional operating system algorithms. Is this practical? Probably not. Is it fun? Absolutely!

## Current Experiments

### Iteration 1: Traditional Algorithms (Baseline)
```
CPU Scheduling:
- Round Robin (RR): Avg completion time: 57.77ms, Throughput: 686,465 ops/s
- Shortest Job First (SJF): Avg completion time: 57.77ms, Throughput: 1,388,842 ops/s

Memory Management:
- LRU: Page faults: 856, Fault rate: 0.856
- FIFO: Page faults: 857, Fault rate: 0.857

Disk Scheduling:
- FCFS: Avg seek time: 64.2ms, Total seek time: 6420ms
- SSTF: Avg seek time: 1.99ms, Total seek time: 199ms
```

### Iteration 2: LLM-based Agents [NEVER WORKED PROPERLY]
T
```
Performance:
- Execution time: ~4.62s (much slower due to API calls)
- CPU: Inconclusive due to high latency
- Memory: Inconclusive due to high latency
- Disk: Inconclusive due to high latency

Key Issues:
- High latency from API calls
- Inconsistent decision making
- Resource intensive
```

### Iteration 3: Deep Q-Network (DQN) Agents
```
CPU Scheduling:
- Avg completion time: -26.5ms (better than traditional)
- Throughput: Still learning to optimize

Memory Management:
- Page fault rate: 19.8% (much better than traditional)
- Better working set management

Disk Scheduling:
- Avg seek time: 43.52ms
- Needs improvement compared to SSTF

Key Improvements:
- Faster execution than LLM
- Better memory management
- Learning from experience
```

### Iteration 4: PPO Agents (Current)
```
CPU Scheduling:
- Avg completion time: -29.08ms :skull_crossbones: (best so far, but no idea why)
- More consistent performance

Memory Management:
- Page fault rate: 19.88%
- Hit rate: 80.12%
- Better adaptation to access patterns

Disk Scheduling:
- Avg seek time: 97.79ms
- Fairness score: Needs improvement
- More balanced between throughput and fairness

Key Improvements:
- More stable learning
- Better handling of trade-offs
- Improved fairness considerations
```


## Current Status

- [x] Traditional algorithms baseline
- [IDK] LLM agents implementation (too slow,interesting, never worked properly, I am broke to afford API calls)
- [x] DQN agents implementation (promising results)
- [x] PPO agents implementation (current best)
- [ ] Hyperparameter optimization
- [ ] Multi-agent coordination
- [ ] Real workload testing

## Next Steps

1. Implement more sophisticated RL architectures
2. Add multi-objective optimization
3. Test with real-world workloads
4. Improve disk scheduling performance
5. Add more metrics and visualizations

## Contributing

Feel free to join in this weird experiment!  Add new AI approaches, Optimize the existing ones, or just tell me why this is a terrible idea, all contributions are welcome.

## License

MIT - Feel free to use this code, but I take no responsibility for any production systems that end up using neural networks for process scheduling 😅
