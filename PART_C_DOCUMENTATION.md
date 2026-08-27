# Part C Documentation — Theory & Reflection Hub

## 1. Purpose
Part C acts as the educational anchor of the Optimizer Lab. It transitions students from empirical observation (Part A and B) to theoretical understanding, ensuring that the behaviors they witnessed are contextualized within formal mathematical and computer science frameworks.

## 2. Theory Sections
The Theory Hub contains structured markdown content explaining the core concepts behind the optimization algorithms.
*(Screenshot: `documentation/screenshots/part-c/part-c-theory.png`)*

## 3. Mathematical Concepts Covered
The hub covers the foundational equations implemented in the backend, primarily:
- Gradient Descent (The base update rule)
- Momentum (Exponential moving averages of velocity)
- Adaptive Learning Rates (Scaling steps by inverse historical gradient magnitude)
- Bias Correction (As seen in Adam)

## 4. Reflection Questions
Interactive accordion components challenge students to synthesize their observations.
*(Screenshot: `documentation/screenshots/part-c/part-c-reflection.png`)*
Examples include:
- "Why does SGD struggle in steep ravines?"
- "How does Adam compare to standard Momentum?"

## 5. Learning Objectives
1. Understand the trade-offs between computational simplicity (SGD) and stateful complexity (Adam).
2. Recognize the symptoms of poor hyperparameter choices (divergence, freezing).
3. Connect 2D geometric navigation to high-dimensional neural network loss landscapes.

## 6. Relationship between Part A observations and Part C theory
Part A allows students to visually confirm the theory explained in Part C. For example, Part C explains that Momentum builds velocity in consistent directions. Part A geometrically visualizes this as the trajectory smoothing out and accelerating down the center of the $L2$ valley, eliminating the perpendicular oscillations of standard SGD.

## 7. Relationship between Part B benchmark results and Part C theory
Part C discusses generalization and overfitting. Part B provides the empirical data for this: students can observe that while adaptive optimizers like Adam might drive Training Loss to zero incredibly fast, they may occasionally exhibit worse generalization (higher Test Loss) than simpler optimizers like SGD with Momentum on certain architectures, validating the theoretical debate around adaptive methods.

## 8. How students are expected to interpret their experimental observations
Students are expected to use Part C as a rubric to grade their experiments. If an optimizer diverges in Part A, the student should consult Part C to understand *why* the mathematical update rule failed (e.g., step size $\eta \cdot g_t$ exceeded the bounds of the local curvature). Part C transforms the lab from a mere visualization toy into a rigorous academic tool.

*(Additional Educational Modals: `documentation/screenshots/part-c/part-c-educational.png`)*
