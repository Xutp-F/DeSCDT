# DeSCDT: Differential Testing Solidity Compiler through Deep Contract Manipulation and Mutation
Solidity, the emergent programming language predominantly utilized for the development of smart
contracts, has been gaining increased importance in the blockchain system. Ensuring the bug-free of
its accompanying language compiler, which is in charge of converting the contract source codes into
executables finally deployed on the blockchain, is thus of paramount importance. Therewith, this study
presents DeSCDT, a differential fuzz testing approach to explore possible defects in the young and
fast evolving solidity compiler. At the core of DeSCDT lies a well-behaving deep contract generator
following the Transformer architecture and learnt with diverse contract code. From an initial seed
pool of smart contracts carefully picked from the wild through semantic encoding and clustering,
the contract generator is capable of stably producing highly syntactic-valid and functional-rich smart
contracts, with three meticulously formulated generation strategies and a set of mutation operations.
Subsequently, in the meantime of compiling these compiler test cases (i.e., the generated contracts)
to trigger obvious compiler crashes, a differential testing environment is setup to explore the potential
defects of the compiler optimizer, by observing the inconsistencies between the expectations and
actual outcomes in the aspects including the gas consumption and the opcode sequence size of the optimized and non-optimized compiled outputs.


## Dataset
The datasets consists of the collected raw information pages of the verified smart contracts on Etherscan, the wild smart contracts deployed on Ethereum blockchain. They are accessbile and can be downloaded from: https://pan.baidu.com/s/17ijVwj0Z2XV0dEQkSonlrQ (Fetch Code: tk5s).

## Source

### Step1:Get model training data
```
python ./genTrainData.py
```

### Step2: Train DeSCDT generative model
```
python ./train.py
```

### Step3: Generate the contract and test the compiler
```
python ./generate.py
```
