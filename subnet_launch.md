## 8. Mint tokens from faucet

You will need tokens to initialize the intentive mechanism on the chain as well as for registering the subnet. 

Run the following commands to mint faucet tokens for the owner and for the validator.

Mint faucet tokens for the owner:

```bash
btcli wallet faucet --wallet.name amitn --subtensor.chain_endpoint ws://127.0.0.1:9944 
```

You will see:

```bash
>> Balance: τ0.000000000 ➡ τ100.000000000
```

Mint tokens for the validator:

```bash
btcli wallet faucet --wallet.name amitn_validator --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see:

```bash
>> Balance: τ0.000000000 ➡ τ100.000000000
```

Mint tokens for the miner:

```bash
btcli wallet faucet --wallet.name amitn_miner --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see:

```bash
>> Balance: τ0.000000000 ➡ τ100.000000000
```

## 9. Create a subnet

The below commands establish a new subnet on the local chain. The cost will be exactly τ100.000000000 for the first subnet you create.

```bash
btcli subnet create --wallet.name amitn --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see:

```bash
>> Your balance is: τ200.000000000
>> Do you want to register a subnet for τ100.000000000? [y/n]: 
>> Enter password to unlock key: [YOUR_PASSWORD]
>> ✅ Registered subnetwork with netuid: 1
```

**NOTE**: The local chain will now have a default `netuid` of 1. The second registration will create a `netuid` 2 and so on, until you reach the subnet limit of 8. If you register more than 8 subnets, then a subnet with the least staked TAO will be replaced by the 9th subnet you register.

## 10. Register keys

Register your subnet validator and subnet miner on the subnet. This gives your two keys unique slots on the subnet. The subnet has a current limit of 128 slots.

Register the subnet miner:

```bash
btcli subnet register --wallet.name amitn_miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9944 --netuid 2
```

Follow the below prompts:

```bash
>> Enter netuid [1] (1): 1
>> Continue Registration? [y/n]: y
>> ✅ Registered
```

Register the subnet validator:

```bash

btcli subnet register --wallet.name amitn_validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9944 --netuid 2
```

Follow the below prompts:

```
>> Enter netuid [1] (1): 1
>> Continue Registration? [y/n]: y
>> ✅ Registered
```

## 11. Add stake 

This step bootstraps the incentives on your new subnet by adding stake into its incentive mechanism.

```bash
btcli stake add --wallet.name amitn_validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9944
```

Follow the below prompts:

```bash
>> Stake all Tao from account: 'validator'? [y/n]: y
>> Stake:
    τ0.000000000 ➡ τ100.000000000
```

## 12. Validate key registrations

Verify that both the miner and validator keys are successfully registered:

```bash
btcli subnet list --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see the `2` entry under `NEURONS` column for the `NETUID` of 1, indicating that you have registered a validator and a miner in this subnet:

```bash
NETUID  NEURONS  MAX_N   DIFFICULTY  TEMPO  CON_REQ  EMISSION  BURN(τ)  
   1        2     256.00   10.00 M    1000    None     0.00%    τ1.00000 
   2      128    
```

See the subnet validator's registered details:

```bash
btcli wallet overview --wallet.name amitn_validator --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see:

```
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  0      True   100.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ100.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0         
```

See the subnet miner's registered details:

```bash
btcli wallet overview --wallet.name amitn_miner --subtensor.chain_endpoint ws://127.0.0.1:9944
```

You will see:

```bash
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  1      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0   

```

## 13. Run subnet miner and subnet validator

Run the subnet miner and subnet validator. Make sure to specify your subnet parameters.

Run the subnet miner:

```bash
python neurons/miner.py --netuid 2 --subtensor.chain_endpoint ws://127.0.0.1:9944 --wallet.name amitn_miner --wallet.hotkey default --logging.debug
```

Run the subnet validator:

```bash
python neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey default --logging.debug
```

## 14. Verify your incentive mechanism

After a few blocks the subnet validator will set weights. This indicates that the incentive mechanism is active. Then after a subnet tempo elapses (360 blocks or 72 minutes) you will see your incentive mechanism beginning to distribute TAO to the subnet miner.

```bash
btcli wallet overview --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946
```

## Ending your session

To halt your nodes:
```bash
# Press CTRL + C keys in the terminal.
```

---