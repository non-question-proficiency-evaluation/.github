# Feasibility and Best Methods

This report analyzes the feasibility of estimating learner proficiency from non-assessment engagement data, evaluates available methodologies, and provides recommendations for matching methods to specific content types. The analysis concludes that while this approach is feasible, it must be treated as an estimation problem under uncertainty, with proficiency estimates considered "lower-confidence evidence" until validated by direct assessment.

## Table of Contents

- [Short Report](#short-report)
  - [1. Achievable?](#1-achievable)
  - [2. Available Methods](#2-available-methods)
  - [3. Recommended Methods](#3-recommended-methods)
  - [4. References](#4-references)
- [Full Report](#full-report)
  - [Executive Summary](#executive-summary)
  - [1. Feasibility Analysis: Is This Achievable?](#1-feasibility-analysis-is-this-achievable)
    - [1.1. Key Supporting Evidence](#11-key-supporting-evidence)
  - [2. Overview of Available Methodologies](#2-overview-of-available-methodologies)
    - [2.1. Heuristic-Based](#21-heuristic-based)
    - [2.2. Model-Based](#22-model-based)
    - [2.3. Machine Learning-Based](#23-machine-learning-based)
  - [3. Recommendation: Matching Methods to Use Cases](#3-recommendation-matching-methods-to-use-cases)
  - [4. Key Risks and Considerations](#4-key-risks-and-considerations)
  - [5. References](#5-references)

## Short Report

### 1. Achievable?

Yes, but it must be treated as an estimation problem under **uncertainty**.

### 2. Available Methods?

| Category | Model Name |
|----------|------------|
| Heuristic-Based | Heuristic Point Systems (XP Models), Time-on-Task & Completion Metrics, Mastery-Based Engagement Scoring (MBES), Time-Weighted Completion Model (TWCM) |
| Model-Based | Engagement-Weighted Bayesian Knowledge Tracing (EW-BKT), Half-Life Regression (HLR), Cognitive Load Proxy Model (CLPM), Performance Factors Analysis (PFA) with Engagement Covariates, Item Response Theory (IRT) Analogy |
| Machine Learning-Based | Deep Knowledge Tracing (DKT), Stealth Assessment (SPRING), Multi-Modal Attention Models (MMAE), Regression/Classification Predictive Models |

### 3. Recommended Methods?

| Job / Content Type | Best Method(s) | Key Signals / Rationale |
|-------------------|----------------|--------------------------|
| Video Content | Cognitive Load Proxy Model (CLPM) | Interprets pauses and rewinds as "germane load" (productive learning) versus fast-forwarding (Guo, P. J., et al., 2014; Yürüm, O. T., et al., 2022). |
| Text / PDF Reading | Half-Life Regression (HLR) | Best for modeling knowledge decay and predicting when a learner will forget information after reading (Settles & Meeder, 2016). |
| Interactive Content | Stealth Assessment (SPRING) | Infers proficiency from action sequences and game logs; validated by Pearson with a correlation of ~0.55 (Gonzalez-Brenes, J. P., et al., 2016). |

### 4. References

* Gonzalez-Brenes, J. P., et al. (2016). A data-driven approach for inferring student proficiency from game activity logs. (Cited: 36)
* Guo, P. J., Kim, J., & Rubin, R. (2014). How video production affects student engagement. (Cited: 3568)
* Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. (Cited: 365)
* Yürüm, O. T., et al. (2022). The use of video clickstream data to predict university students' test performance. (Cited: 32)

## Full Report

### Executive Summary

Estimating Learner Proficiency from Engagement Data

### 1. Feasibility Analysis: Is This Achievable?

This report addresses the core question of whether it is possible to reliably estimate learning proficiency from non-assessment interactions, such as watching videos, reading content, or engaging with simulations. The following analysis provides a direct, evidence-based answer, detailing the statistical foundation for this approach and the necessary conditions for its successful implementation.

The definitive judgment is that this approach is **feasible**, but it must be treated as an estimation problem under **uncertainty**. While traditional assessments provide clear signals of mastery, engagement data offers "rich" but indirect evidence. **Therefore, proficiency estimates derived from engagement should be considered "lower-confidence evidence" until they are validated by direct assessment.**

#### 1.1. Key Supporting Evidence

The feasibility of this approach is supported by consistent findings across academic research and commercial applications:

* Statistical Correlation: Engagement-based proficiency estimation can achieve a moderate to strong correlation with formal assessment outcomes, with studies consistently reporting correlation coefficients in the range of r = 0.40 to 0.65.
* Behavioral Indicators: Specific, fine-grained user behaviors are highly predictive of learning. For example, pausing and rewinding videos are positively correlated with higher exam performance, whereas frequent fast-forwarding is associated with lower scores (Guo, P. J., et al., 2014; Yürüm, O. T., et al., 2022).
* Commercial Validation: Pearson's SPRING model provides a powerful proof of concept. This "stealth assessment" system successfully predicts student test outcomes from game logs with a correlation of ~0.55, demonstrating that structured interaction sequences can serve as a valid proxy for proficiency (Gonzalez-Brenes, J. P., et al., 2016).

Major learning platforms track engagement data extensively but maintain a generally conservative stance on its use for proficiency claims. Khan Academy, for instance, is primarily assessment-centric and does not count watching videos or reading articles toward its "mastery" system. Similarly, platforms like Brilliant and Codecademy emphasize "learning by doing" by immediately following content with interactive challenges. Duolingo uses its Half-Life Regression model to estimate memory strength from practice attempts, not passive reading (Settles & Meeder, 2016). MOOC platforms like Coursera and edX use engagement analytics primarily to flag at-risk learners rather than to award proficiency scores. This industry-wide caution underscores that engagement is treated as a support activity, not direct proof of learning.

While feasibility is well-established, the success of this approach depends entirely on selecting the right methodology. We now turn to an overview of the available methods for translating engagement into proficiency estimates.

### 2. Overview of Available Methodologies

A variety of methods exist to translate raw engagement data into meaningful proficiency estimates. These approaches range from simple, rule-based heuristics to complex, data-intensive neural networks, each suited for different analytical purposes and operational contexts. This section categorizes and evaluates the primary methodologies available.

#### 2.1. Heuristic-Based

* Heuristic Point Systems (XP Models): Assigns experience points or progress percentages for completing content. (Strength: Simple and interpretable for gamification; Weakness: Lacks statistical rigor).
* Time-on-Task & Completion Metrics: Uses normalized time spent and content completion rates as direct predictors of success. (Strength: Easy to calculate and understand; Weakness: Does not account for the quality or efficiency of engagement).
* Mastery-Based Engagement Scoring (MBES): Assigns proficiency levels (e.g., "Familiar," "Proficient") based on pre-defined engagement triggers. (Strength: Provides clear, categorical labels; Weakness: Thresholds are arbitrary and not statistically derived).
* Time-Weighted Completion Model (TWCM): Weights content completion by the quality of time spent relative to an expected duration. (Strength: Adds a layer of nuance to simple completion; Weakness: Still relies on expert-defined rules).

#### 2.2. Model-Based

* Engagement-Weighted Bayesian Knowledge Tracing (EW-BKT): Treats content interactions as learning opportunities that probabilistically update a learner's skill mastery state (Corbett & Anderson, 1994). (Strength: Theoretically grounded in cognitive science; Weakness: Assumes a binary learned/unlearned state for each skill).
* Half-Life Regression (HLR): Models the Ebbinghaus forgetting curve to estimate memory strength and predict the probability of recall over time (Settles & Meeder, 2016). (Strength: Excellent for modeling knowledge decay and spaced repetition; Weakness: Requires data on practice attempts to be most effective).
* Cognitive Load Proxy Model (CLPM): Estimates productive learning effort ("germane load") by analyzing interaction patterns like video pauses and rewinds (Guo, P. J., et al., 2014; Yürüm, O. T., et al., 2022). (Strength: Directly infers cognitive engagement from behavior; Weakness: Interpretation of signals can be context-dependent).
* Performance Factors Analysis (PFA) with Engagement Covariates: A logistic regression model that incorporates both prior performance and current engagement to predict proficiency. (Strength: Integrates prior performance for more accurate prediction; Weakness: Dependent on the availability of historical performance data).
* Item Response Theory (IRT) Analogy: An adaptation that treats content pieces as "items" of varying difficulty to validate proficiency gain based on future performance. (Strength: Provides a psychometrically robust framework for content validation; Weakness: Relies on an analogy and requires downstream assessment data).

#### 2.3. Machine Learning-Based

* Deep Knowledge Tracing (DKT): Uses Recurrent Neural Networks to process complex sequences of student interactions to predict future performance (Piech, C., et al., 2015). (Strength: Highly predictive and captures non-linear patterns; Weakness: Complex, requires large datasets, and can be a "black box").
* Stealth Assessment (e.g., Pearson's SPRING): A data-driven pipeline that infers proficiency from action sequences and game logs without direct questioning (Gonzalez-Brenes, J. P., et al., 2016). (Strength: Provides assessment without interrupting the learning flow; Weakness: Requires a structured design framework like Evidence-Centered Design).
* Multi-Modal Attention Models (MMAE): Combines disparate signals like scroll depth and playback speed to infer attention quality and learning. (Strength: Synthesizes multiple weak signals into a stronger inference; Weakness: Computationally intensive and can be difficult to interpret).
* Regression/Classification Predictive Models: Direct models like Random Forests trained to predict final outcomes from early "clickstream" features. (Strength: Straightforward to implement and highly predictive for specific outcomes; Weakness: Can lack the explanatory power of cognitive models).

Having reviewed the landscape of available methods, the next step is to align these powerful tools with specific analytical tasks to ensure their effective application.

### 3. Recommendation: Matching Methods to Use Cases

The strategic value of estimating proficiency from engagement lies in selecting the right method for the right task. The choice of methodology should be driven by the type of learning content being analyzed and the specific insights required. This section provides a clear, actionable guide for applying the most effective approach for common learning scenarios.

The table below summarizes the best methods for evaluating proficiency based on the learning content type, detailing the key behavioral signals each method leverages.

Table 1: Recommended Methodologies by Content Type

| Job / Content Type | Best Method(s) | Key Signals / Rationale |
|-------------------|----------------|--------------------------|
| Video Content | Cognitive Load Proxy Model (CLPM) | Interprets pauses and rewinds as "germane load" (productive learning) versus fast-forwarding (Guo, P. J., et al., 2014; Yürüm, O. T., et al., 2022). |
| Text / PDF Reading | Half-Life Regression (HLR) | Best for modeling knowledge decay and predicting when a learner will forget information after reading (Settles & Meeder, 2016). |
| Interactive Content | Stealth Assessment (SPRING) | Infers proficiency from action sequences and game logs; validated by Pearson with a correlation of ~0.55 (Gonzalez-Brenes, J. P., et al., 2016). |


The optimal method is context-dependent, and this table serves as a primary decision-making tool for matching analytical needs with proven methodologies. This strategic alignment is critical for navigating the inherent challenges of the approach.

### 4. Key Risks and Considerations

While estimating proficiency from engagement is feasible and valuable, it is not without challenges. A successful implementation requires careful management of the primary risks and unknown variables associated with interpreting indirect evidence of learning.

1. Uncertainty of Evidence: The foremost consideration is that engagement data provides "lower-confidence evidence" of learning compared to direct assessment. It should be treated as an unconfirmed estimate that signals a probability of proficiency, not a guarantee. These estimates must eventually be validated by performance on assessment tasks.
2. Conservatism of Industry Leaders: The cautious approach of major platforms suggests that relying solely on engagement data for high-stakes proficiency claims is not yet a standard or validated industry practice. This conservatism reflects the inherent difficulty in proving mastery without direct evidence of application.
3. Passive vs. Active Learning: Underlying this entire approach is the cognitive principle that passive study (e.g., watching a video) is inherently less predictive of long-term retention than active practice and problem-solving. This finding is supported by established cognitive science frameworks like ICAP (Chi & Wylie, 2014), which link different modes of engagement to varying learning outcomes. Models must account for this distinction to avoid overestimating proficiency from low-effort interactions.

Ultimately, the approach is promising and actionable, provided that robust, model-based methods are used to navigate the uncertainty of evidence and account for the cognitive differences between active and passive learning.

### 5. References

* Chi, M. T. H., & Wylie, R. (2014). The ICAP framework: Linking cognitive engagement to active learning outcomes. (Cited: 3912)
* Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. (Cited: 3370 )
* Gonzalez-Brenes, J. P., et al. (2016). A data-driven approach for inferring student proficiency from game activity logs. (Cited: 36)
* Guo, P. J., Kim, J., & Rubin, R. (2014). How video production affects student engagement. (Cited: 3568)
* Piech, C., et al. (2015). Deep knowledge tracing. (Cited: 3568)
* Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. (Cited: 365)
* Tong, X., & Ren, Y. (2025). Deep knowledge tracing and cognitive load estimation for personalized learning path. (Cited: 5)
* Yürüm, O. T., et al. (2022). The use of video clickstream data to predict university students' test performance. (Cited: 32)
