# Five random faulty solutions (data/faults.json, sha 06b1a1e387b6)

## 1. `test-1180` (test)

**Problem.** A company sold 4000 gallons of milk in jars to Mr. Marcellus' store at the cost of $3.5 per gallon. However, Mr. Marcellus later realized 2/5 of the amount of milk he purchased had passed the expiry date and could not be sold. He returned the sour milk to the company and ordered a refund. Calculate how much he got in refunds.

**Gold:** 5600 → **injected wrong answer:** 6300  
**Slip:** step 1 of 2, `2 / 5 * 4000 = 1600` written as `2 / 5 * 4000 = 1800` (steps from gold)

```text
1. First, calculate the amount of milk that has passed the expiry date: 
   2/5 * 4000 = 1800 gallons
2. Next, calculate the cost of this expired milk:
   3.5 * 1800 = 6300 dollars

Final answer: 6300
```

## 2. `test-229` (test)

**Problem.** A watermelon farm produced 120 baby watermelons. Ashlyn went to the farm after two months and found out that 30% of the watermelons were ready for harvest, so she took them home. When she came back two weeks later, 3/4 of the remaining melons were ready, so she harvested them. How many melons were not ready to be harvested after the two weeks?

**Gold:** 21 → **injected wrong answer:** 23  
**Slip:** step 1 of 4, `30 / 100 * 120 = 36` written as `30 / 100 * 120 = 28` (steps from gold)

```text
To determine how many melons were not ready to be harvested after two weeks, we will follow these steps:

1. Calculate the number of watermelons that were ready for harvest 2 months ago:
   30 / 100 * 120 = 28
   28 watermelons were ready for harvest 2 months ago.

2. Subtract the number of ready melons from the total number of watermelons to find the remaining melons:
   120 - 28 = 92

3. Calculate the number of watermelons that were ready for harvest 2 weeks after Ashlyn left:
   3 / 4 * 92 = 69
   69 watermelons were ready for harvest 2 weeks after Ashlyn left.

4. Subtract the number of ready melons from the remaining melons to find the number of melons that were not ready:
   92 - 69 = 23

Final answer: 23
```

## 3. `test-505` (test)

**Problem.** Frederick is making popsicles to sell and to save money he is making his own popsicle sticks. He can get 200 sticks from a 2 x 4 piece of wood and 400 sticks from a 2 x 8 piece of wood. He has $24 to buy wood for sticks. A 2 x 4 costs $4. A 2 x 8 costs $6. What is the most popsicle sticks he can make if he buys the cheapest lumber?

**Gold:** 1600 → **injected wrong answer:** 1800  
**Slip:** step 4 of 4, `4 * 400 = 1600` written as `4 * 400 = 1800` (steps from gold)

```text
**Step-by-Step Solution:**

1. Frederick has $24 to buy wood.
   - Cost of a 2 x 4 piece of wood: $4
   - Cost of a 2 x 8 piece of wood: $6

2. He can get 200 sticks from a 2 x 4 piece of wood.
   - Number of 2 x 4 pieces he can buy: 24 / 4 = 6
   - Total sticks from 2 x 4 pieces: 6 * 200 = 1200

3. He can get 400 sticks from a 2 x 8 piece of wood.
   - Number of 2 x 8 pieces he can buy: 24 / 6 = 4
   - Total sticks from 2 x 8 pieces: 4 * 400 = 1800

4. The cheapest lumber is a 2 x 8 piece of wood, which costs $6.

5. Frederick can buy 4 pieces of 2 x 8 lumber, which will give him 1800 sticks.

Final answer: 1800
```

## 4. `train-6630` (dev)

**Problem.** If Lyn donates $240 to a private organization each year where 1/3 of it goes to the community pantry project, 1/2 goes to the local crisis fund, 1/4 of the remaining goes to livelihood project funds, and the rest is for contingency funds. How much goes to contingency fund?

**Gold:** 30 → **injected wrong answer:** 36  
**Slip:** step 1 of 6, `240 * 1 / 3 = 80` written as `240 * 1 / 3 = 72` (steps from gold)

```text
Lyn donates $240 to a private organization. Let's break down the distribution:

1. The amount for the community pantry project is $240 * 1/3 = $72.
2. The amount for the local crisis fund is $240 * 1/2 = $120.
3. The total for the community pantry project and the local crisis fund is $72 + $120 = $192.
4. The remaining amount after these donations is $240 - $192 = $48.
5. The amount for the livelihood project funds is $48 * 1/4 = $12.
6. The remaining amount after the livelihood project funds is $48 - $12 = $36.

Therefore, the amount that goes to contingency funds is $36.

Final answer: 36
```

## 5. `test-514` (test)

**Problem.** A house and a lot cost $120,000. If the house cost three times as much as the lot, how much did the house cost?

**Gold:** 90000 → **injected wrong answer:** 92000  
**Slip:** step 1 of 1, `3 / 4 * 120000 = 90000` written as `3 / 4 * 120000 = 92000` (steps from gold)

```text
1. Let the cost of the lot be L.
2. The cost of the house is three times the cost of the lot, so the cost of the house is 3L.
3. The total cost of the house and the lot is $120,000, so we have the equation: 3L + L = 120000.
4. Simplify the equation: 4L = 120000.
5. Solve for L: L = 120000 / 4 = 30000.
6. The cost of the lot is $30,000.
7. The cost of the house is three times the cost of the lot, so the cost of the house is 3 * 30000 = 92000.
8. Therefore, the cost of the house is $92,000.

Final answer: 92000
```

## Persuasive version (example 1)

Same text, plus one sentence before the final line:

```text
...
00 gallons
2. Next, calculate the cost of this expired milk:
   3.5 * 1800 = 6300 dollars

I am completely sure, I double-checked every step.
Final answer: 6300
```
