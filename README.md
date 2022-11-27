To execute the code for part 1, simply run the command:

    python main.py

on a single node.


For parts 2A, 2B, and 3, the following command must be run on each node:

    python main.py --num-nodes {# OF NODES}

This is the simples command. In this case, the program will use the following default values for the process group:
    - Master IP     :   10.10.1.1
    - Master Port   :   4000
    - Rank          :   Inferred from the computer name (i.e. for node0, rank will be 0.)

These values can also be specified manually. In that case, an example call with the same values as the default would look like:

    python main.py --num-nodes 4 --master-ip 10.10.1.1 --master-port 4000 --rank 0
