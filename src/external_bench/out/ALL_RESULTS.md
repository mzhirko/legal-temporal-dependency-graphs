```
== ALL RUNS ==
run                               n acc/exact  abst infra  extra
airline_hard_gemma               80       0.0    53     ?  w1%=0 w10%=4
airline_hard_llama               80       0.0     7     ?  w1%=1 w10%=7
housing_hard_gemma               78    0.2821    11     ?  
housing_hard_llama               78    0.3205     0     ?  
lextime_llm_gemma               200      0.71     0     0  
lextime_llm_llama               200      0.79     0     0  
lextime_tdg_gemma               200     0.005   199     0  
sara_binary_hard_gemma           30    0.5667     0     ?  
sara_binary_hard_llama           30    0.4667     1     ?  
sara_numeric_hard_gemma          35    0.0857     1     ?  w1%=11 w10%=20
sara_numeric_hard_llama          35       0.0     3     ?  w1%=2 w10%=10
smoke_sara                        5       0.0     5     ?  
smoke_sara_llama                  5       0.8     0     ?  
smoke_tracie_llm                 10       0.7     0     ?  
smoke_tracie_tdg                 10       0.0    10     ?  
tracie_llm_gemma                200      0.76     0     ?  
tracie_llm_llama                200      0.64     1     0  
uscis_hard_gemma                 28    0.3214    11     ?  
uscis_hard_llama                 28    0.4643     1     ?  

== LEXTIME BY PAIR TYPE ==
lextime_llm_gemma          both-explicit              n=105  acc_answered=0.7905 acc_all=0.7905 abst=0
lextime_llm_gemma          both-implicit              n=3    acc_answered=0.6667 acc_all=0.6667 abst=0
lextime_llm_gemma          one-implicit-one-explicit  n=92   acc_answered=0.6196 acc_all=0.6196 abst=0
lextime_llm_llama          both-explicit              n=105  acc_answered=0.8 acc_all=0.8 abst=0
lextime_llm_llama          both-implicit              n=3    acc_answered=0.3333 acc_all=0.3333 abst=0
lextime_llm_llama          one-implicit-one-explicit  n=92   acc_answered=0.7935 acc_all=0.7935 abst=0
lextime_tdg_gemma          both-explicit              n=105  acc_answered=1.0 acc_all=0.0095 abst=104
lextime_tdg_gemma          both-implicit              n=3    acc_answered=None acc_all=0.0 abst=3
lextime_tdg_gemma          one-implicit-one-explicit  n=92   acc_answered=None acc_all=0.0 abst=92

== BINARY CONFUSION (gold -> pred) ==
housing_hard_gemma             NO->NO:13  NO->YES:23  YES->NO:22  YES->YES:9
housing_hard_llama             NO->NO:10  NO->YES:29  YES->NO:24  YES->YES:15
lextime_llm_gemma              YES->NO:58  YES->YES:142
lextime_llm_llama              YES->NO:42  YES->YES:158
lextime_tdg_gemma              YES->YES:1
sara_binary_hard_gemma         NO->NO:11  NO->YES:5  YES->NO:8  YES->YES:6
sara_binary_hard_llama         NO->NO:7  NO->YES:9  YES->NO:6  YES->YES:7
smoke_sara_llama               NO->NO:3  YES->NO:1  YES->YES:1
smoke_tracie_llm               NO->NO:5  YES->NO:3  YES->YES:2
tracie_llm_gemma               NO->NO:93  NO->YES:7  YES->NO:41  YES->YES:59
tracie_llm_llama               NO->NO:64  NO->YES:36  YES->NO:35  YES->YES:64
uscis_hard_gemma               ACCEPTED->ACCEPTED:5  ACCEPTED->DISMISSED:2  DISMISSED->ACCEPTED:6  DISMISSED->DISMISSED:4
uscis_hard_llama               ACCEPTED->ACCEPTED:10  ACCEPTED->DISMISSED:3  DISMISSED->ACCEPTED:11  DISMISSED->DISMISSED:3

== GOLD BALANCE (majority-class baseline) ==
housing_hard_gemma             {'YES': 39, 'NO': 39}  majority-baseline=0.50
housing_hard_llama             {'YES': 39, 'NO': 39}  majority-baseline=0.50
lextime_llm_gemma              {'YES': 200}  majority-baseline=1.00
lextime_llm_llama              {'YES': 200}  majority-baseline=1.00
lextime_tdg_gemma              {'YES': 200}  majority-baseline=1.00
sara_binary_hard_gemma         {'YES': 14, 'NO': 16}  majority-baseline=0.53
sara_binary_hard_llama         {'YES': 14, 'NO': 16}  majority-baseline=0.53
smoke_sara                     {'YES': 2, 'NO': 3}  majority-baseline=0.60
smoke_sara_llama               {'YES': 2, 'NO': 3}  majority-baseline=0.60
smoke_tracie_llm               {'YES': 5, 'NO': 5}  majority-baseline=0.50
smoke_tracie_tdg               {'YES': 5, 'NO': 5}  majority-baseline=0.50
tracie_llm_gemma               {'YES': 100, 'NO': 100}  majority-baseline=0.50
tracie_llm_llama               {'YES': 100, 'NO': 100}  majority-baseline=0.50
uscis_hard_gemma               {'ACCEPTED': 14, 'DISMISSED': 14}  majority-baseline=0.50
uscis_hard_llama               {'ACCEPTED': 14, 'DISMISSED': 14}  majority-baseline=0.50
```
