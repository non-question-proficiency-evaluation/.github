# non-question-proficiency-evaluation

NEXS-399_non-question-proficiency-evaluation

This document contains the project structure and presentation questions for the Non-Question Proficiency Evaluation Framework (NEXS-399). It includes key questions organized by category that need to be answered for the project presentation, covering feasibility, methods, technical details, validation, implementation, practical examples, and strategic considerations.

## Table of Contents

- [1. Project Structure](#1-project-structure)
- [2. Presentation Questions](#2-presentation-questions)
  - [2.1. Fundamental Questions (Feasibility & Foundation)](#21-fundamental-questions-feasibility--foundation)
    - [2.1.1. Is estimating proficiency gain from non-question learning activities feasible?](#211-is-estimating-proficiency-gain-from-non-question-learning-activities-feasible)
    - [2.1.2. What methods exist for estimating proficiency gain from engagement data?](#212-what-methods-exist-for-estimating-proficiency-gain-from-engagement-data)
    - [2.1.3. Which method is best for the Non-Question Proficiency Evaluation Framework?](#213-which-method-is-best-for-the-non-question-proficiency-evaluation-framework)
  - [2.2. Technical Questions (Technical Details)](#22-technical-questions-technical-details)
    - [2.2.1. How do proficiency gain estimation methods work?](#221-how-do-proficiency-gain-estimation-methods-work)
    - [2.2.2. What engagement metrics are needed to estimate proficiency gain?](#222-what-engagement-metrics-are-needed-to-estimate-proficiency-gain)
    - [2.2.3. How accurate are proficiency gain estimates from engagement data?](#223-how-accurate-are-proficiency-gain-estimates-from-engagement-data)
  - [2.3. Validation Questions (Validation)](#23-validation-questions-validation)
    - [2.3.1. How can proficiency gain estimation models be validated?](#231-how-can-proficiency-gain-estimation-models-be-validated)
    - [2.3.2. What are the challenges and limitations of estimating proficiency without questions?](#232-what-are-the-challenges-and-limitations-of-estimating-proficiency-without-questions)
  - [2.4. Implementation Questions (Implementation)](#24-implementation-questions-implementation)
    - [2.4.1. How should the proficiency evaluation framework be implemented?](#241-how-should-the-proficiency-evaluation-framework-be-implemented)
    - [2.4.2. What data is needed to implement the proficiency gain estimation system?](#242-what-data-is-needed-to-implement-the-proficiency-gain-estimation-system)
  - [2.5. Practical Questions (Practical)](#25-practical-questions-practical)
    - [2.5.1. What are practical examples of proficiency gain estimation?](#251-what-are-practical-examples-of-proficiency-gain-estimation)
    - [2.5.2. How do major learning platforms estimate proficiency from non-assessment activities?](#252-how-do-major-learning-platforms-estimate-proficiency-from-non-assessment-activities)
  - [2.6. Strategic Questions (Strategic)](#26-strategic-questions-strategic)
    - [2.6.1. Which proficiency estimation method should be prioritized for implementation?](#261-which-proficiency-estimation-method-should-be-prioritized-for-implementation)
    - [2.6.2. Is the proficiency evaluation framework scalable?](#262-is-the-proficiency-evaluation-framework-scalable)
    - [2.6.3. Should proficiency gain estimates be personalized for individual students?](#263-should-proficiency-gain-estimates-be-personalized-for-individual-students)

## 1. Project Structure

- **`Deep Research/`** - Research sources organized into `0. Archive/`, `1. Deep Research/`, and `2. Comparison/`
- **`Prompts/`** - Prompt files and templates for research and synthesis tasks

## 2. Presentation Questions

This section contains key questions that need to be answered for the project presentation, organized by category.

### 2.1. Fundamental Questions (Feasibility & Foundation)

#### 2.1.1. Is estimating proficiency gain from non-question learning activities feasible?

**Answer:** Feasibility of Estimating Proficiency from Non-Question Activities

**Estimating proficiency gain from non-question activities (such as reading, watching videos, or interactive simulations) is considered feasible, though it is viewed as an estimation problem under uncertainty.** While traditional assessments provide clear "right or wrong" signals, non-question interactions provide "rich engagement data" that can be used to infer learning through probabilistic models. Modern frameworks suggest that every digital interaction potentially signals something about a student's learning state. However, sources note that because passive study is less predictive of retention than active practice, these estimates should often be treated as **"lower-confidence evidence"** or "unconfirmed" until validated by a later assessment.

#### 2.1.1.1. Research and Empirical Evidence

**Research and empirical evidence strongly support the correlation between meaningful content engagement and learning outcomes.** Key findings include:
*   **Engagement Correlations:** Studies consistently show that higher engagement—measured by time spent, completion rates, and re-engagement—correlates with better test scores. For instance, every additional minute on Khan Academy was associated with gains on standardized tests.
*   **Behavioral Indicators:** Fine-grained behaviors like **pausing and rewinding videos** are positively correlated with higher exam performance, whereas frequent fast-forwarding is associated with lower performance.
*   **Stealth Assessment:** Pearson's **SPRING** (Student PRoficiency INferrer from Game data) research demonstrated that a data-driven pipeline could predict test outcomes from game logs with a **correlation of approximately 0.55**, validating that action sequences can predict learning without direct quiz questions.
*   **Predictive Power:** Research in MOOCs found that attendance and utilization rates in the first week of a course could highly predict which students would eventually pass.
*   **Statistical Accuracy:** Engagement-based proficiency estimation can achieve a moderate to strong correlation with assessment outcomes, typically ranging from **r = 0.40 to 0.65**.

#### 2.1.1.2. Practices of Major Platforms

**Most major learning platforms track engagement data extensively but are conservative about using it as the sole proof of proficiency.**
*   **Khan Academy:** Their mastery system is primarily assessment-centric; watching videos or reading articles **does not directly count toward "mastery"**. They treat videos as support tools, prioritizing "learning by doing" to ensure high confidence in mastery.
*   **Duolingo:** While Duolingo uses XP (Experience Points) for gamification, their core proficiency model, **Half-Life Regression (HLR)**, historically relies on practice attempts rather than just reading lesson tips. However, they use this data to model **forgetting curves** and memory strength over time.
*   **Coursera and edX:** These platforms measure achievement through assignment grades and course completion. While they do not provide a numeric proficiency score based solely on views, they use video analytics and engagement scores to **flag at-risk learners** or recommend content.
*   **Brilliant and Codecademy:** These platforms emphasize "learning by doing" and typically follow up content consumption with an immediate interactive challenge or coding task to measure gain.

#### 2.1.1.3. Evidence of Success with This Approach

**Evidence of success is found primarily in predictive accuracy, improved user retention, and the ability to personalize learning paths.**
*   **Improved Prediction:** Duolingo's HLR model, which transforms behavior data into proficiency estimates, improved the prediction of word recall and increased **daily user retention by 12%**.
*   **Early Intervention:** MOOC research shows that using engagement metrics (like video "utilization rate") early in a course allows for automated "nudges" that successfully help students stay on track.
*   **Cognitive Load Balancing:** Recent studies (2025) integrating knowledge tracing with cognitive load estimation (based on engagement patterns) have led to **more efficient personalized learning paths**.
*   **Validation of Learning Gains:** Success has also been measured by comparing "stealth" estimates with external standardized tests (like MAP Growth), showing that students identified as highly engaged by the models do indeed show significant growth norms.

***

Estimating proficiency from content engagement is like tracking a hiker's progress by observing their pace and the terrain they cover; while you can reasonably estimate how far they've come, you don't know for certain they've reached the summit until they check in at the peak.

#### 2.1.2. What methods exist for estimating proficiency gain from engagement data?

**Answer:** Based on the sources provided, there are several distinct methods for estimating proficiency gains from engagement data, ranging from simple rule-based systems to complex neural networks. These methods are categorized into heuristic, model-based, and machine-learning-based approaches.

#### 2.1.2.1. Heuristic-Based Methods

These methods rely on pre-defined rules and expert judgment rather than statistical inference. They are often used for immediate feedback and gamification.

*   **Heuristic Point Systems (XP Models):** Assigns experience points or progress percentages for completing content (e.g., Duolingo's XP or Khan Academy's energy points). 
*   **Time-on-Task & Completion Metrics:** Uses normalized time spent and completion rates (e.g., percentage of a video watched) as direct predictors of success. 
*   **Mastery-Based Engagement Scoring (MBES):** A threshold-based system where proficiency levels (Attempted, Familiar, Proficient, Mastered) are assigned based on engagement triggers.
*   **Time-Weighted Completion Model (TWCM):** A simple model that weights content completion by the quality of time spent relative to expected duration.

#### 2.1.2.2. Model-Based (Probabilistic and Statistical) Methods

These methods use established psychological or cognitive theories to model how knowledge is acquired or forgotten over time.

*   **Engagement-Weighted Bayesian Knowledge Tracing (EW-BKT):** An extension of standard BKT that treats content interactions as "learning opportunities." It uses engagement signals (completion, interaction density) to modify the probability that a student has transitioned from an unlearned to a learned state.
*   **Half-Life Regression (HLR):** A model that combines the **Ebbinghaus forgetting curve** with engagement data to estimate the "strength" of a learner's memory and predict the probability of recall over time.
*   **Performance Factors Analysis (PFA) with Engagement Covariates:** A logistic regression model that incorporates both a student's prior performance and current engagement factors to predict proficiency.
*   **Item Response Theory (IRT) Analogy:** While typically used for questions, it can be adapted to treat content pieces as "items" with specific difficulty levels, where future success on related questions validates the proficiency gained from that content.
*   **Cognitive Load Proxy Model (CLPM):** Estimates the "germane load" (productive learning effort) versus "extraneous load" by analyzing engagement patterns like video pauses and rewinds.

#### 2.1.2.3. Machine Learning (ML)-Based Methods

These data-driven approaches learn complex, non-linear mappings between behavioral logs and proficiency outcomes.

*   **Deep Knowledge Tracing (DKT) with Engagement Features:** Uses Recurrent Neural Networks (RNNs) or Transformers to process sequences of student interactions (including non-question data) to predict future performance.
*   **Stealth Assessment (e.g., Pearson's SPRING):** A data-driven pipeline that uses **Evidence-Centered Design (ECD)** to infer proficiency from action sequences and game logs without direct questioning.
*   **Multi-Modal Attention Models (MMAE):** Combines multiple disparate signals—such as scroll depth, video playback speed changes, and session frequency—to infer the quality of attention and subsequent learning.
*   **Regression/Classification Predictive Models:** Direct models (Random Forests, Logistic Regression) trained to predict final exam scores or mastery states early in a course based on "clickstream" features.

---

#### 2.1.2.4. References for Key Methods (Peer-Reviewed)

The sources identify the following core academic references for these methodologies:

| Method | Primary Peer-Reviewed Reference(s) |
| :--- | :--- |
| **Bayesian Knowledge Tracing** | Corbett, A. T., & Anderson, J. R. (1994). *Knowledge tracing: Modeling the acquisition of procedural knowledge.* |
| **Half-Life Regression** | Settles, B., & Meeder, B. (2016). *A trainable spaced repetition model.* |
| **Stealth Assessment (SPRING)** | Gonzalez-Brenes et al. (2016). *A Data-Driven Approach for Inferring Student Proficiency from Game Activity Logs.* |
| **Deep Knowledge Tracing** | Piech, C., et al. (2015). *Deep knowledge tracing.* |
| **Video Engagement Analytics** | Guo, P. J., Kim, J., & Rubin, R. (2014). *How video production affects student engagement.* |
| **Video Clickstream Prediction** | Yürüm et al. (2022). *The use of video clickstream data to predict university students' test performance.* |
| **DKT + Cognitive Load** | Tong & Ren (2025). *Deep knowledge tracing and cognitive load estimation for personalized learning path.* |
| **ICAP Framework** | Chi, M. T. H., & Wylie, R. (2014). *The ICAP framework: Linking cognitive engagement to active learning outcomes.* |

***

**Analogy:** If learning is a journey, **Heuristics** are like counting the steps taken; **Model-based** approaches are like using a map and known walking speeds to estimate location; and **ML-based** approaches are like using a satellite to analyze every subtle movement and terrain change to predict exactly when the traveler will arrive.

#### 2.1.3. Which method is best for the Non-Question Proficiency Evaluation Framework?

**Answer:** The following table provides a short summary of the recommended methods for estimating proficiency from non-question engagement data, categorized by the specific learning "job" or content type.

### Summary of Best Methods for Proficiency Evaluation

| Job / Content Type | Best Method(s) | Key Signals / Rationale |
| :--- | :--- | :--- |
| **Video Content** | **Cognitive Load Proxy Model (CLPM)** | Interprets **pauses and rewinds** as "germane load" (productive learning) versus fast-forwarding. |
| **Text / PDF Reading** | **Half-Life Regression (HLR)** | Best for modeling **knowledge decay** and predicting when a learner will forget information after reading. |
| **Interactive Content** | **Stealth Assessment (SPRING)** | Infers proficiency from **action sequences** and game logs; validated by Pearson with a correlation of ~0.55. |
| **Memory Retention** | **Half-Life Regression (HLR)** | Superior for **spaced practice**; incorporates forgetting curves to maintain realistic proficiency scores over time. |
| **General Skill Core** | **Engagement-Weighted BKT (EW-BKT)** | Acts as a probabilistic **"backbone"** to update skill states whenever a learning opportunity (content interaction) occurs. |
| **Sequence Analysis** | **Deep Knowledge Tracing (DKT)** | Captures complex, **non-linear temporal patterns** in interaction sequences to predict future assessment performance. |

***

### References
Based on the sources, the following peer-reviewed works underpin these methods:

*   **Corbett, A. T., & Anderson, J. R. (1994).** *Knowledge tracing: Modeling the acquisition of procedural knowledge.*
*   **Settles, B., & Meeder, B. (2016).** *A trainable spaced repetition model for language learning.*
*   **Gonzalez-Brenes, J. P., et al. (2016).** *A data-driven approach for inferring student proficiency from game activity logs.*
*   **Piech, C., et al. (2015).** *Deep knowledge tracing.*
*   **Guo, P. J., Kim, J., & Rubin, R. (2014).** *How video production affects student engagement.*
*   **Yürüm, O. T., et al. (2022).** *The use of video clickstream data to predict university students’ test performance.*
*   **Tong, X., & Ren, Y. (2025).** *Deep knowledge tracing and cognitive load estimation for personalized learning path.*
*   **Chi, M. T. H., & Wylie, R. (2014).** *The ICAP framework: Linking cognitive engagement to active learning outcomes.*

### 2.2. Technical Questions (Technical Details)

#### 2.2.1. How do proficiency gain estimation methods work?

**Answer:** _[To be filled]_

#### 2.2.2. What engagement metrics are needed to estimate proficiency gain?

**Answer:** _[To be filled]_

#### 2.2.3. How accurate are proficiency gain estimates from engagement data?

**Answer:** _[To be filled]_

### 2.3. Validation Questions (Validation)

#### 2.3.1. How can proficiency gain estimation models be validated?

**Answer:** _[To be filled]_

#### 2.3.2. What are the challenges and limitations of estimating proficiency without questions?

**Answer:** _[To be filled]_

### 2.4. Implementation Questions (Implementation)

#### 2.4.1. How should the proficiency evaluation framework be implemented?

**Answer:** _[To be filled]_

#### 2.4.2. What data is needed to implement the proficiency gain estimation system?

**Answer:** _[To be filled]_

### 2.5. Practical Questions (Practical)

#### 2.5.1. What are practical examples of proficiency gain estimation?

**Answer:** _[To be filled]_

#### 2.5.2. How do major learning platforms estimate proficiency from non-assessment activities?

**Answer:** _[To be filled]_

### 2.6. Strategic Questions (Strategic)

#### 2.6.1. Which proficiency estimation method should be prioritized for implementation?

**Answer:** _[To be filled]_

#### 2.6.2. Is the proficiency evaluation framework scalable?

**Answer:** _[To be filled]_

#### 2.6.3. Should proficiency gain estimates be personalized for individual students?

**Answer:** _[To be filled]_
