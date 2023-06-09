# Part 1: Backpropagation

### Formula section: having formulas for backpropagation, basically gradient calculation in back propagation

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/3f4f00e6-25fe-46c5-9311-a6051d82bf52)

  - 1: all the formulas
  - 2: detotal/dw5 complete formula opening term by term
  - 3: total loss partial derivative wrt to hidden layer to output layer connection weights
  - 4: de_total/da_h1: going back one more layer so that later can find derivative of loss wrt to input layer to hidden layer connecting weights.
  - 5: opening terms for de_total/da_h1
  - 6: final formual for de_total/da_h1
  - 7: opening terms for calculating derivative of loss wrt input layer to hidden layer connecting weights
  - 8: final formulas for initial weights loss gradient

### Now going in the same flow, calculate gradients wrt to all weights and update weights using learning rate
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/f67ad11f-c213-42ac-be01-1f980c5af051)


### Losses when:
1. LR = 0.1: 
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/a9eaf0a5-2ff3-43b5-bf7a-80710466fd89)

2. LR = 0.2:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/100263f6-99e3-4982-9a4a-8991aa511baa)

3. LR = 0.5:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/5d588bba-23d8-4a2e-908c-efd1f752c76f)

4. LR = 0.8:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/3f58cb7c-09d6-4cee-b25e-4cfea10e4ec4)

5. LR = 1:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/89adb12c-6a7f-4210-b914-0fef876f7a96)

6. LR = 2:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/7974ecf0-17f5-40b9-bba6-7ab309d06052)

#### Observation: Loss going down faster as we are increasing learning rate.


# Part 2:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/bd03950c-542f-42e4-8d06-a7fe3320afb9)

Reached accuracy of 99.42 in 20th epoch with total 18738 parameters


