Corrected UGV01 capacity-contract simulation
=============================================

Extract this ZIP into the DigitalTwinDivergence repository root. It creates:

  fordesk/capacity_contract_simulator.py
  fordesk/plot_simulator.py

PowerShell commands from the repository root:

  python fordesk\capacity_contract_simulator.py `
    --repo-root . `
    --output-dir capacity_sim_results

  python fordesk\plot_simulator.py `
    --input capacity_sim_results\capacity_contract_summary.csv `
    --output capacity_sim_results\capacity_contract_simulation.png

Optional capacity selection:

  --capacities 1.5,2,5,10,20

The plot defaults to the global_state_tracking service. To plot another one,
append one of these to the plotting command:

  --service local_1s_tight
  --service local_5s_moderate
  --service local_10s_preview

Important interpretation
------------------------

This is a trace-driven cached-source replay. It preserves the source timestamps
and never creates fresh measurements above the recorded source rate. Therefore,
it is suitable for transport-capacity sensitivity, retransmission analysis, and
position-only contract replay. It is not evidence of measured 5/10/20 Hz sensor
fidelity or measured communication savings.
