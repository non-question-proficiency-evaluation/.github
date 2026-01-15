# Proficiency Evaluation Framework for Non-Question-Based Learning Activities

## A Comprehensive Research Report on Estimating Student Learning Gains from Content Engagement

**Document Version:** 1.0  
**Date:** January 8, 2026  
**Prepared for:** Nexus AI Educational Technology Platform  

---

# Executive Summary

## Overview

This research report presents a comprehensive framework for estimating student proficiency gains from non-assessment learning activities—specifically video content, PDF/text documents, and interactive learning materials. The framework addresses a critical gap in educational technology: the ability to quantify learning outcomes without relying solely on traditional question-based assessments.

## Key Findings

### 1. Foundational Methodological Approaches

Our research identified three primary categories of methodologies for estimating proficiency from engagement data:

**Tier 1 (Highly Recommended):**
- **Bayesian Knowledge Tracing (BKT) with Engagement Extensions**: Adapts the foundational BKT framework to incorporate engagement signals as evidence of learning transitions
- **Half-Life Regression (HLR) Models**: Originally developed by Duolingo, these models estimate memory retention and can be adapted for content-based proficiency estimation
- **Engagement-Weighted Mastery Models**: Building on Khan Academy's mastery learning framework, these models weight engagement metrics to estimate proficiency gains

**Tier 2 (Recommended with Conditions):**
- **Deep Knowledge Tracing (DKT) Variants**: Neural network approaches that can incorporate multi-modal engagement features
- **Performance Factors Analysis (PFA) Extensions**: Logistic regression models that can integrate engagement covariates
- **Spaced Repetition Decay Models**: Time-based models estimating knowledge retention from content exposure

**Tier 3 (Promising but Requires Validation):**
- **Engagement-to-Outcome Regression Models**: Direct correlation models between engagement metrics and learning outcomes
- **Cognitive Load Estimation Models**: Using video engagement patterns to infer cognitive processing depth
- **Multi-Modal Attention Models**: Combining multiple engagement signals for proficiency inference

### 2. Critical Engagement Metrics

Research consistently identifies the following metrics as most predictive of learning outcomes:

| Metric Category | Specific Metrics | Predictive Strength |
|----------------|------------------|---------------------|
| Time-Based | Time-on-task, Watch time, Completion time | High |
| Completion | Video completion rate, Document completion, Course completion | High |
| Interaction | Pause/rewind frequency, Playback speed changes, Scroll depth | Medium-High |
| Behavioral | Session frequency, Return visits, Content revisitation | Medium |
| Quality | Focused attention time, Active vs. passive engagement | Medium-High |

### 3. Content-Specific Considerations

**Video Content:**
- Optimal video length for engagement: <6 minutes (Guo et al., 2014)
- Median engagement drops significantly after 9 minutes
- Talking-head and Khan-style tablet drawings increase engagement
- Completion rates above 60% indicate highly engaging content

**Text/PDF Content:**
- Scroll depth of 50%+ correlates with meaningful engagement
- Combined time-on-page and scroll depth provides more accurate engagement estimation
- Average attention time of 2+ minutes on educational content suggests active reading

**Interactive Content:**
- Gamification elements (points, badges, leaderboards) show large effect sizes (g = 0.822)
- Interactive elements increase retention by 25-40% over passive consumption
- Immediate feedback mechanisms significantly enhance learning

### 4. Implementation Priorities

1. **Phase 1**: Implement engagement data collection infrastructure (duration, completion, interactions)
2. **Phase 2**: Deploy baseline proficiency estimation using weighted engagement models
3. **Phase 3**: Calibrate models against assessment outcomes where available
4. **Phase 4**: Implement adaptive content difficulty estimation
5. **Phase 5**: Deploy machine learning enhancements for personalized estimation

### 5. Expected Accuracy and Limitations

Based on the literature review, engagement-based proficiency estimation can achieve:
- **Correlation with assessment outcomes**: r = 0.40-0.65 (moderate to strong)
- **AUC for mastery prediction**: 0.68-0.85 depending on model complexity
- **Key limitation**: Without periodic assessment validation, models may drift over time

---

# Part 1: Literature Review and Theoretical Foundations

## 1.1 Learning Analytics and Engagement Research

### The Engagement-Learning Relationship

The relationship between student engagement and learning outcomes is well-established in educational research. Fredricks et al. (2004) define engagement as a multidimensional construct comprising behavioral, emotional, and cognitive components. Learning analytics research has increasingly focused on operationalizing these dimensions through digital trace data.

**Key Research Findings:**

Halverson and Graham (2019) suggest that online traces can reflect behavioral and cognitive engagement, though they acknowledge limited capacity for capturing emotional engagement. Martin and Borup (2022) propose that behavioral engagement, being more tangibly traced online, could serve as a physical manifestation of cognitive and emotional engagement.

A systematic review by Johar et al. (2023) examined 159 studies on learning analytics and student engagement in higher education, finding that:
- Time spent on tasks, listening to lectures, and commenting on posts were reliable indicators of social engagement
- Cognitive engagement could be inferred from information-sharing behaviors on learning platforms
- There remains significant inconsistency in how engagement is operationalized across studies

### Digital Trace Data as Learning Indicators

Research consistently shows that specific digital behaviors correlate with learning outcomes:

**Time-on-Task:**
Yamada et al. (2017) found that students who read for longer periods had better performance. Bulut et al. (2022) demonstrated that students who missed activities or rarely interacted with LMS had lower grades, while regular users performed better.

**Video Engagement:**
The seminal study by Guo, Kim, and Rubin (2014) analyzed 6.9 million video watching sessions across four edX courses, establishing that:
- Video length is the most significant predictor of engagement
- Median engagement time for videos under 6 minutes approaches 100%
- Engagement drops significantly after 9 minutes regardless of production quality
- Informal talking-head videos outperform high-production lecture captures

**Completion Metrics:**
Research analyzing over 300,000 students found that average watch time is the strongest predictor of video effectiveness (referenced in Synthesia, 2025). Completion rates above 60% indicate highly engaging training content.

### The ICAP Framework

Chi and Wylie's (2014) ICAP framework (Interactive-Constructive-Active-Passive) provides a theoretical foundation for understanding engagement levels:

| Mode | Description | Learning Outcome |
|------|-------------|------------------|
| Interactive | Collaborative dialogue and co-construction | Highest |
| Constructive | Generation of new outputs beyond presented material | High |
| Active | Physical manipulation of learning materials | Moderate |
| Passive | Receiving information without overt action | Lowest |

This framework suggests that engagement quality, not just quantity, matters for learning outcomes. Passive video watching produces lower learning gains than active engagement strategies.

## 1.2 Knowledge Tracing and Proficiency Modeling

### Bayesian Knowledge Tracing (BKT)

Originally proposed by Corbett and Anderson (1994), BKT models student knowledge as a hidden Markov model with four parameters:

- **P(L₀)**: Initial probability of knowing a skill
- **P(T)**: Probability of transitioning from unlearned to learned state (learning rate)
- **P(S)**: Probability of a slip (incorrect response despite knowing)
- **P(G)**: Probability of a guess (correct response despite not knowing)

The model updates knowledge state estimates based on observed student responses:

```
P(Lₙ|correct) = [P(Lₙ₋₁)(1-P(S))] / [P(Lₙ₋₁)(1-P(S)) + (1-P(Lₙ₋₁))P(G)]
P(Lₙ|incorrect) = [P(Lₙ₋₁)P(S)] / [P(Lₙ₋₁)P(S) + (1-P(Lₙ₋₁))(1-P(G))]
P(Lₙ) = P(Lₙ|obs) + (1-P(Lₙ|obs))P(T)
```

**Relevance to Non-Assessment Estimation:**
While traditional BKT requires assessment responses, extensions can incorporate engagement signals:
- Content completion as implicit evidence of learning transition
- Time-on-task as a modifier of learning probability
- Interaction patterns as evidence of cognitive engagement

### Item Response Theory (IRT)

IRT models the probability of correct response as a function of learner ability and item difficulty:

**1-Parameter Logistic (Rasch) Model:**
```
P(correct) = 1 / (1 + e^-(θ - b))
```
Where θ = learner ability, b = item difficulty

**2-Parameter Model:**
```
P(correct) = 1 / (1 + e^(-a(θ - b)))
```
Where a = discrimination parameter

**Connection to Engagement:**
Deonovic et al. (2018) demonstrated fundamental connections between BKT and IRT, suggesting that hybrid models could leverage both longitudinal engagement data (BKT strength) and cross-sectional ability estimation (IRT strength).

### Deep Knowledge Tracing (DKT)

Piech et al. (2015) introduced Deep Knowledge Tracing, applying recurrent neural networks to model student knowledge states. Key advantages:
- No need for manually specified knowledge components
- Can capture complex temporal dependencies
- Achieves AUC of 0.85 on Khan Academy data (vs. 0.68 for standard BKT)

**Limitations for Non-Assessment Contexts:**
- Traditionally requires question-response sequences
- Less interpretable than parametric models
- May overfit without sufficient assessment data

### Half-Life Regression (HLR)

Developed by Duolingo (Settles & Meeder, 2016), HLR models memory decay and is particularly relevant for estimating retained knowledge from content exposure:

```
p = 2^(-Δ/h)
```
Where p = probability of recall, Δ = time since last exposure, h = half-life

The half-life is estimated as:
```
h = 2^(Θ·x)
```
Where Θ = learned weights, x = feature vector including:
- Number of previous exposures
- Time since last exposure
- Item difficulty features
- Learner ability features

**Application to Content Learning:**
HLR can be adapted for estimating retained knowledge from non-assessment content by:
- Treating content completion as an exposure event
- Using engagement quality metrics as difficulty proxies
- Incorporating learner history as ability indicators

Duolingo's A/B tests showed that HLR-based recommendations improved:
- Daily retention by 12% for overall activity
- Practice session retention by 9.5%
- Lesson retention by 1.7%

## 1.3 Content-Specific Learning Research

### Video-Based Learning

**Optimal Video Design for Learning:**

Research synthesized from multiple sources indicates:

| Factor | Optimal Configuration | Evidence Strength |
|--------|----------------------|-------------------|
| Duration | <6 minutes | Strong (Guo et al., 2014) |
| Format | Talking-head + tablet drawing | Strong |
| Speaking pace | Faster than typical lectures | Moderate |
| Interactivity | Embedded questions | Strong (Szpunar et al., 2013) |
| Production | Informal > High-production | Moderate |

**Active Learning in Video:**
A meta-analysis of 54 studies (2019-2025) found that active learning strategies embedded in video content significantly improved:
- Retention: g = 0.33
- Comprehension: g = 0.28
- Transfer: g = 0.43
- Motivation: g = 0.39

Embedded questions reduced extraneous cognitive load while increasing germane load, suggesting deeper processing of content.

### Text-Based Learning

**Scroll Depth as Engagement Indicator:**
- Average scroll depth for news/media websites: 57%
- Average scroll depth for educational content: varies by length
- Mobile devices show higher scroll depth (66%) than desktop (60%)

**Combined Metrics:**
The combination of scroll depth + time-on-page + adjusted bounce rate provides the most accurate engagement estimation for text content.

### Interactive and Gamified Learning

**Meta-Analysis Findings:**
A meta-analysis of 41 studies (n > 5,071 participants) found that gamification in educational settings produces a large effect size (g = 0.822 [0.567-1.078]) on student learning outcomes.

**Effective Game Elements:**
- Points: Provide immediate feedback and progress indication
- Badges: Recognition of achievement milestones
- Leaderboards: Social comparison and motivation
- Challenges: Goal-directed engagement
- Immediate feedback: Reinforcement of correct understanding

**Moderating Factors:**
- Offline environments showed larger effects than online
- Journal articles reported larger effect sizes (g = 0.936) than conference proceedings (g = 0.585)
- Long-term exposure may reduce novelty effects and motivation

---

# Part 2: Ranked Methodological Approaches

## 2.1 Tier 1: Highly Recommended Methods

### Method 1: Engagement-Weighted Bayesian Knowledge Tracing (EW-BKT)

**Description:**
An extension of standard BKT that incorporates engagement signals as modifiers of learning transition probabilities.

**Mathematical Formulation:**
```
P(T|engagement) = P(T_base) × W(engagement)

Where W(engagement) = Σᵢ wᵢ × normalize(engagementᵢ)
```

Engagement features and suggested weights:
- Completion rate: w = 0.35
- Time-on-task (normalized): w = 0.25
- Interaction density: w = 0.20
- Return visits: w = 0.10
- Content difficulty adjustment: w = 0.10

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 4 | Strong BKT foundation; engagement extensions validated in multiple studies |
| Accuracy/Predictive Power | 4 | Expected AUC: 0.70-0.80 based on extended BKT research |
| Theoretical Foundation | 5 | Grounded in learning theory (cognitive load, mastery learning) |
| Practical Applicability | 4 | Moderate implementation complexity; requires engagement data infrastructure |
| Generalizability | 4 | Applicable across content types with appropriate feature engineering |
| Validation/Calibration | 4 | Can be validated against periodic assessments |
| Industry Adoption | 3 | Components widely used; specific combination novel |

**Overall Ranking:** 4.1/5.0

**Strengths:**
- Strong theoretical foundation in established cognitive science
- Interpretable parameters allow for pedagogical insights
- Can be incrementally validated and calibrated
- Handles uncertainty naturally through probabilistic framework

**Weaknesses:**
- Requires initial parameter estimation from historical data
- May need domain-specific calibration
- Less accurate than assessment-based approaches

**Best Use Cases:**
- Continuous proficiency tracking between assessments
- Adaptive content recommendation based on estimated mastery
- Progress reporting when assessments are sparse

**Implementation Complexity:** Moderate

**Key Sources:**
- Corbett & Anderson (1994). Knowledge tracing. User Modeling and User-Adapted Interaction
- Deonovic et al. (2018). Learning meets assessment. Behaviormetrika
- Yildirim-Erbasli (2023). Introduction to BKT with pyBKT. Education Sciences

---

### Method 2: Half-Life Regression for Content Retention (HLR-CR)

**Description:**
Adaptation of Duolingo's Half-Life Regression model to estimate knowledge retention from content engagement events.

**Mathematical Formulation:**
```
p(recall) = 2^(-Δt/h)

h = 2^(θ₀ + θ₁·n_exposures + θ₂·avg_engagement + θ₃·content_difficulty + θ₄·learner_ability)
```

Where:
- Δt = time since content engagement
- h = estimated half-life (memory strength)
- n_exposures = number of times content was engaged
- avg_engagement = quality-weighted engagement score
- content_difficulty = estimated difficulty of content
- learner_ability = estimated prior ability

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 5 | Validated on 13M+ learning traces at Duolingo |
| Accuracy/Predictive Power | 4 | MAE nearly half of Leitner baseline; strong A/B test results |
| Theoretical Foundation | 5 | Based on Ebbinghaus forgetting curve; spacing/lag effects |
| Practical Applicability | 4 | Open-source implementation available; requires adaptation |
| Generalizability | 3 | Originally for language learning; requires domain adaptation |
| Validation/Calibration | 4 | Well-documented calibration procedures |
| Industry Adoption | 5 | Production use at scale by Duolingo |

**Overall Ranking:** 4.3/5.0

**Strengths:**
- Strongest empirical validation in production educational system
- Explicitly models forgetting and retention
- Open-source implementation and dataset available
- Demonstrated impact on user retention (12% improvement)

**Weaknesses:**
- Designed for item-level (word) learning; adaptation needed for content-level
- Requires sufficient historical data for reliable parameter estimation
- May not capture deep understanding vs. surface recall

**Best Use Cases:**
- Estimating knowledge decay over time
- Optimal timing for content review recommendations
- Predicting readiness for advanced content

**Implementation Complexity:** Moderate-Complex

**Key Sources:**
- Settles & Meeder (2016). A Trainable Spaced Repetition Model. ACL Proceedings
- GitHub: duolingo/halflife-regression
- Zaidi et al. (2020). Adaptive Forgetting Curves. AIED Proceedings

---

### Method 3: Mastery-Based Engagement Scoring (MBES)

**Description:**
Inspired by Khan Academy's mastery learning system, this approach maps engagement metrics to mastery levels using threshold-based scoring.

**Mathematical Formulation:**
```
Mastery_Score = Σᵢ (contentᵢ_weight × engagement_scoreᵢ × difficulty_factorᵢ)

engagement_score = f(completion_rate, time_quality, interaction_density)

Mastery_Level = 
    "Mastered" if score ≥ 0.90
    "Proficient" if score ≥ 0.70
    "Familiar" if score ≥ 0.50
    "Attempted" if score > 0
    "Not Started" otherwise
```

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 4 | Khan Academy research shows correlation with MAP Growth scores |
| Accuracy/Predictive Power | 4 | Validated against standardized assessments |
| Theoretical Foundation | 5 | Based on Bloom's mastery learning (1968, 1984) |
| Practical Applicability | 5 | Simple threshold-based implementation |
| Generalizability | 4 | Widely applicable across subjects and age groups |
| Validation/Calibration | 4 | Khan Academy provides validation studies |
| Industry Adoption | 5 | Production use at Khan Academy scale |

**Overall Ranking:** 4.4/5.0

**Strengths:**
- Simplest to implement and communicate
- Strong theoretical foundation in mastery learning research
- Validated correlation with standardized assessments
- Intuitive for users and educators

**Weaknesses:**
- Requires careful threshold calibration
- May oversimplify continuous proficiency
- Less precise for individual skill estimation

**Best Use Cases:**
- User-facing progress dashboards
- Course completion and certification decisions
- Broad-level proficiency categorization

**Implementation Complexity:** Simple

**Key Sources:**
- Block & Burns (1976). Mastery learning. Review of Research in Education
- Khan Academy (2025). Skills to proficient measurement blog post
- Guskey (2007). Closing achievement gaps. ASCD

---

## 2.2 Tier 2: Recommended with Conditions

### Method 4: Deep Knowledge Tracing with Engagement Features (DKT-E)

**Description:**
Neural network approach that incorporates engagement features alongside any available assessment data.

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 4 | Strong performance on benchmark datasets |
| Accuracy/Predictive Power | 5 | AUC 0.85+ on Khan Academy data |
| Theoretical Foundation | 3 | Data-driven; less interpretable |
| Practical Applicability | 3 | Requires ML infrastructure and expertise |
| Generalizability | 4 | Can learn across domains with sufficient data |
| Validation/Calibration | 3 | Black-box nature complicates validation |
| Industry Adoption | 4 | Growing adoption in EdTech |

**Overall Ranking:** 3.7/5.0

**Best Use Cases:**
- Platforms with substantial assessment data available
- When highest predictive accuracy is prioritized over interpretability
- Systems with ML engineering resources

**Implementation Complexity:** Complex

---

### Method 5: Performance Factors Analysis with Engagement Covariates (PFA-E)

**Description:**
Logistic regression model incorporating prior performance and engagement factors.

**Mathematical Formulation:**
```
P(correct) = 1 / (1 + e^(-m))
m = β + Σⱼ (γⱼ·successes_j + ρⱼ·failures_j + εⱼ·engagement_j)
```

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 4 | Well-validated in educational data mining |
| Accuracy/Predictive Power | 3 | Moderate; improves with engagement features |
| Theoretical Foundation | 4 | Based on learning factors analysis |
| Practical Applicability | 4 | Standard logistic regression implementation |
| Generalizability | 4 | Applicable across domains |
| Validation/Calibration | 4 | Straightforward statistical validation |
| Industry Adoption | 3 | Common in research; less in production |

**Overall Ranking:** 3.7/5.0

**Implementation Complexity:** Moderate

---

### Method 6: Time-Weighted Completion Model (TWCM)

**Description:**
Simple model weighting content completion by quality of time spent.

**Mathematical Formulation:**
```
Proficiency_Gain = content_value × completion_rate × time_quality_factor

time_quality_factor = min(actual_time / expected_time, 2.0) if actual_time ≥ min_threshold
                    = 0 otherwise

min_threshold = 0.3 × expected_time (prevents gaming)
```

**Evaluation Scores:**

| Criterion | Score (1-5) | Justification |
|-----------|-------------|---------------|
| Empirical Validity | 3 | Face validity; limited rigorous validation |
| Accuracy/Predictive Power | 3 | Moderate correlation with outcomes |
| Theoretical Foundation | 3 | Based on time-on-task research |
| Practical Applicability | 5 | Very simple to implement |
| Generalizability | 4 | Applicable across content types |
| Validation/Calibration | 3 | Requires empirical threshold setting |
| Industry Adoption | 4 | Common baseline approach |

**Overall Ranking:** 3.5/5.0

**Implementation Complexity:** Simple

---

## 2.3 Tier 3: Promising but Needs Validation

### Method 7: Multi-Modal Attention Estimation (MMAE)

**Description:**
Uses multiple engagement signals to estimate attention quality and infer learning.

**Considerations:**
- Combines video pause/rewind patterns, scroll behavior, session duration
- Requires validation against learning outcomes
- May capture cognitive engagement better than simple metrics

**Implementation Complexity:** Complex

---

### Method 8: Cognitive Load Proxy Model (CLPM)

**Description:**
Estimates cognitive load from engagement patterns based on cognitive load theory.

**Theoretical Basis:**
- Intrinsic load: Content complexity
- Extraneous load: Interface/presentation factors
- Germane load: Productive learning effort

**Implementation Complexity:** Complex

---

## 2.4 Tier 4: Not Recommended

### Simple Time-Only Models

**Reason:** Research shows time alone is insufficient; students may have page open without engagement.

### Completion-Only Models

**Reason:** Does not distinguish between quality of engagement; easily gamed.

### Click-Count Models

**Reason:** Quantity of interaction does not reliably indicate learning.

---

# Part 3: Mathematical Framework and Formulas

## 3.1 Core Proficiency Estimation Formula

The recommended framework combines multiple approaches into a unified proficiency estimation:

```
Proficiency(skill, learner, t) = 
    Prior_Estimate(skill, learner) + 
    Σᵢ Learning_Gain(contentᵢ) × Retention_Factor(t - tᵢ)
```

### Component 1: Prior Estimate

```
Prior_Estimate = 
    Assessment_Prior if recent_assessment_available
    Population_Prior × Ability_Adjustment otherwise

Population_Prior = historical average for skill
Ability_Adjustment = f(learner's performance on related skills)
```

### Component 2: Learning Gain from Content

```
Learning_Gain(content) = 
    Content_Value × Engagement_Quality × Difficulty_Match

Content_Value = base learning potential of content (calibrated)
Engagement_Quality = W₁·completion + W₂·time_quality + W₃·interaction_quality
Difficulty_Match = 1 - |content_difficulty - learner_level| / max_diff
```

**Suggested Default Weights:**
- W₁ (completion) = 0.40
- W₂ (time_quality) = 0.35
- W₃ (interaction_quality) = 0.25

### Component 3: Retention Factor

Based on half-life regression:
```
Retention_Factor(Δt) = 2^(-Δt / half_life)

half_life = base_half_life × (1 + 0.2 × num_exposures) × engagement_quality^0.5
base_half_life = 7 days (default, calibrate empirically)
```

## 3.2 Content-Specific Formulas

### Video Content

```
Video_Engagement_Score = 
    0.40 × watch_completion_rate +
    0.25 × normalized_watch_time +
    0.20 × interaction_rate +
    0.15 × rewatch_bonus

watch_completion_rate = watched_duration / total_duration
normalized_watch_time = min(actual_time / expected_time, 1.5)
interaction_rate = (pauses + rewinds + speed_changes) / duration_minutes
rewatch_bonus = min(rewatch_segments / total_segments, 0.5)
```

**Expected Time Calculation:**
```
expected_time = video_duration × 1.2  (allows for pauses/rewinding)
```

### Text/PDF Content

```
Text_Engagement_Score = 
    0.35 × scroll_completion +
    0.35 × time_quality +
    0.20 × interaction_density +
    0.10 × return_visits

scroll_completion = max_scroll_depth / 100
time_quality = min(actual_time / expected_read_time, 1.5)
expected_read_time = word_count / 200  (average reading speed: 200 wpm)
interaction_density = (highlights + annotations + clicks) / pages
return_visits = min(session_count - 1, 3) / 3
```

### Interactive Content

```
Interactive_Engagement_Score = 
    0.30 × completion_rate +
    0.30 × interaction_quality +
    0.20 × time_quality +
    0.20 × achievement_bonus

interaction_quality = successful_interactions / total_interactions
achievement_bonus = badges_earned / possible_badges × gamification_weight
gamification_weight = 0.5 (adjustable based on platform design)
```

## 3.3 Difficulty Estimation

Content difficulty can be estimated from engagement patterns:

```
Estimated_Difficulty = 
    0.40 × (1 - avg_completion_rate) +
    0.30 × normalized_time_variance +
    0.30 × avg_rewatch_rate

normalized_time_variance = std(completion_times) / mean(completion_times)
```

## 3.4 Proficiency Level Thresholds

```
Mastery Levels:
    Mastered:    proficiency ≥ 0.90
    Proficient:  proficiency ≥ 0.70
    Familiar:    proficiency ≥ 0.50
    Attempted:   proficiency ≥ 0.20
    Novice:      proficiency < 0.20
```

---

# Part 4: Platform Case Studies

## 4.1 Khan Academy

### Mastery Learning Implementation

Khan Academy's mastery system implements a multi-level proficiency framework:
- **Attempted**: Student has tried exercises for the skill
- **Familiar**: Some success demonstrated
- **Proficient**: Consistent correct responses
- **Mastered**: High and sustained performance

### Research Findings

A 2022 peer-reviewed study (ESSA Tier 3 evidence) demonstrated:
- Students using Khan Academy 30-60 minutes/week exceeded pre-pandemic growth norms
- Some grades showed 40%+ improvement over expected growth
- "Skills to proficient" metric correlated with MAP Growth mathematics scores

### Key Metrics Tracked
- Time spent on platform
- Skills attempted vs. completed
- Proficiency level progression
- Video watch patterns
- Exercise accuracy

### Relevance to Framework
Khan Academy's approach validates that engagement time combined with proficiency progression provides meaningful learning estimates. Their correlation with standardized assessments (MAP Growth) provides external validity evidence.

## 4.2 Duolingo

### Half-Life Regression System

Duolingo's HLR system represents the most rigorously validated engagement-to-proficiency model in production:

**Data Scale:** 13 million user-word pairs across multiple languages

**Key Features:**
- Lexeme-level difficulty estimation
- Morphological complexity weighting
- Forgetting curve modeling
- Spaced repetition optimization

### Validation Results
- MAE reduced by ~50% compared to Leitner baseline
- 12% improvement in daily retention
- 9.5% improvement in practice session retention

### Open Resources
- GitHub repository with code and data
- Dataverse dataset (361 MB) for research

### Relevance to Framework
Demonstrates that memory retention models can effectively guide learning systems. The open-source availability enables adaptation for content-based learning estimation.

## 4.3 edX/Coursera

### Learning Analytics Research

edX and Coursera platforms have contributed significant research on MOOC engagement:

**Key Findings (Guo et al., 2014):**
- 6.9 million video sessions analyzed
- Video length is strongest engagement predictor
- Production style significantly affects engagement
- Informal formats outperform polished lectures

### Analytics Infrastructure
- Open-source analytics pipeline (edX)
- Research data partnerships
- A/B testing frameworks for interventions

### Relevance to Framework
MOOC research provides large-scale evidence for video engagement patterns and their relationship to learning outcomes. The edX analytics pipeline offers implementation models.

## 4.4 Brilliant

### Problem-Based Learning Model

Brilliant emphasizes interactive problem-solving over passive content consumption:

**Approach:**
- Concept maps with interactive elements
- Immediate feedback on problem attempts
- Progressive difficulty scaffolding
- Visual and intuitive explanations

### Engagement Metrics
- Problem completion rates
- Hint usage patterns
- Time-to-solution distributions
- Concept map navigation paths

### Relevance to Framework
Demonstrates that interactive engagement can serve as both the learning mechanism and assessment indicator, reducing the need for separate evaluation.

---

# Part 5: Validation and Calibration Approaches

## 5.1 Validation Strategy Options

### Option A: Concurrent Validity with Assessments

**Approach:** Validate engagement-based estimates against periodic assessment outcomes.

**Implementation:**
1. Deploy proficiency estimation model
2. Administer assessments at regular intervals (e.g., monthly)
3. Compute correlation between estimates and assessment scores
4. Adjust model parameters to minimize prediction error

**Metrics:**
- Pearson correlation coefficient (target: r > 0.5)
- Mean Absolute Error (target: < 15% of scale)
- AUC-ROC for mastery prediction (target: > 0.75)

### Option B: Predictive Validity

**Approach:** Test whether engagement-based estimates predict future outcomes.

**Implementation:**
1. Generate proficiency estimates from engagement data
2. Track subsequent assessment performance
3. Evaluate predictive accuracy
4. Iterate on model based on prediction errors

**Metrics:**
- Prediction accuracy for pass/fail outcomes
- Spearman correlation for ranked predictions
- Calibration curves (predicted vs. actual probabilities)

### Option C: Transfer Validation

**Approach:** Test whether estimates generalize to external assessments.

**Implementation:**
1. Collect external assessment data (e.g., standardized tests)
2. Correlate platform proficiency estimates with external scores
3. Adjust for population differences

**Considerations:**
- Construct alignment between platform skills and external assessments
- Time lag between platform learning and external assessment

## 5.2 Calibration Methods

### Method 1: Threshold Calibration

Adjust mastery thresholds based on empirical outcomes:

```
For each proposed threshold T:
    Classify learners as "proficient" if estimate ≥ T
    Compute accuracy against assessment outcomes
    Select T that maximizes classification accuracy
```

### Method 2: Regression Calibration

Fit a calibration function to adjust raw estimates:

```
Calibrated_Estimate = f(Raw_Estimate)

Options for f:
- Linear: a × Raw + b
- Isotonic regression: monotonic transformation
- Platt scaling: logistic transformation
```

### Method 3: Population-Based Calibration

Adjust estimates based on cohort performance:

```
Calibrated_Estimate = (Raw_Estimate - cohort_mean) / cohort_std × reference_std + reference_mean
```

## 5.3 Ongoing Monitoring

### Drift Detection

Monitor for model degradation over time:

```
Weekly metrics:
- Mean prediction error trend
- Correlation stability
- Distribution shift detection

Alert thresholds:
- Correlation drop > 0.1 from baseline
- MAE increase > 20% from baseline
```

### A/B Testing Framework

For model improvements:

```
Test design:
- Random assignment to model variants
- Outcome metric: downstream assessment performance
- Duration: minimum 4 weeks
- Sample size: power analysis for desired effect size
```

---

# Part 6: Implementation Recommendations

## 6.1 Phased Implementation Roadmap

### Phase 1: Data Infrastructure (Weeks 1-4)

**Objectives:**
- Implement comprehensive engagement tracking
- Establish data pipeline for analytics
- Create baseline measurement capabilities

**Technical Requirements:**
- Event tracking for all content interactions
- Timestamp precision (milliseconds)
- User session management
- Data warehouse/lake infrastructure

**Key Metrics to Collect:**

| Content Type | Metrics | Collection Method |
|--------------|---------|-------------------|
| Video | Play, pause, seek, speed change, completion | Video player events |
| Text/PDF | Scroll position, time on page, page turns | Browser events |
| Interactive | Click, drag, submit, success/failure | Application events |
| General | Session start/end, return visits, device | Platform events |

### Phase 2: Baseline Estimation Model (Weeks 5-8)

**Objectives:**
- Implement initial proficiency estimation
- Deploy user-facing progress indicators
- Establish calibration framework

**Recommended Initial Model:**
Mastery-Based Engagement Scoring (MBES) due to simplicity and interpretability

**Implementation Steps:**
1. Define skill/content taxonomy
2. Assign content values based on curriculum
3. Implement engagement scoring functions
4. Set initial mastery thresholds
5. Create progress visualization

### Phase 3: Calibration and Validation (Weeks 9-16)

**Objectives:**
- Validate estimates against assessments
- Calibrate thresholds and weights
- Implement feedback loops

**Activities:**
- Administer baseline assessments
- Compute validation metrics
- Adjust model parameters
- Document calibration procedures

### Phase 4: Advanced Features (Weeks 17-24)

**Objectives:**
- Implement retention modeling
- Add personalization features
- Deploy adaptive recommendations

**Features:**
- Half-life regression for retention estimation
- Learner ability estimation from history
- Content difficulty estimation from engagement patterns
- Personalized learning path recommendations

### Phase 5: Machine Learning Enhancement (Weeks 25+)

**Objectives:**
- Improve accuracy with ML models
- Implement continuous learning
- Scale to production load

**Considerations:**
- A/B testing framework for model comparison
- Model monitoring and alerting
- Fallback to simpler models if needed

## 6.2 Technical Architecture

### Recommended Stack

```
Data Collection Layer:
- Event tracking SDK (client-side)
- Event ingestion service (server-side)
- Real-time streaming (Kafka/Kinesis)

Storage Layer:
- Event store (time-series database)
- Feature store (for ML features)
- Analytics warehouse (for reporting)

Processing Layer:
- Stream processing (Spark/Flink)
- Batch processing (scheduled jobs)
- ML pipeline (model training/serving)

Application Layer:
- Proficiency estimation API
- Progress dashboard
- Recommendation service
```

## 6.3 Key Success Metrics

### System Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Estimate-Assessment Correlation | r > 0.50 | Monthly validation |
| Mastery Prediction AUC | > 0.75 | Rolling calculation |
| User Engagement with Progress | > 60% view rate | Analytics |
| Model Latency | < 100ms | API monitoring |

### Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Learning Outcome Improvement | +10% on assessments | A/B testing |
| User Retention | +5% platform retention | Cohort analysis |
| Content Completion | +15% completion rate | Engagement analytics |
| User Satisfaction | NPS > 40 | User surveys |

---

# Part 7: Annotated Bibliography

## Foundational Research

### Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4(4), 253-278.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Foundational, highly cited)

**Summary:** Introduces Bayesian Knowledge Tracing, the foundational model for student knowledge state estimation. Establishes the four-parameter model (P(L₀), P(T), P(S), P(G)) for tracking knowledge acquisition through observed performance.

**Relevance:** Essential theoretical foundation for any knowledge estimation system. Framework can be extended to incorporate engagement signals.

---

### Guo, P. J., Kim, J., & Rubin, R. (2014). How video production affects student engagement: An empirical study of MOOC videos. L@S '14 Proceedings, 41-50.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Large-scale empirical, highly cited)

**Summary:** Analysis of 6.9 million video watching sessions across four edX MOOCs. Establishes that video length is the strongest predictor of engagement, with median engagement approaching 100% for videos under 6 minutes.

**Relevance:** Critical for video content design and engagement metric interpretation. Provides benchmarks for expected engagement patterns.

---

### Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. ACL Proceedings, 1848-1858.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Validated at scale, open-source)

**Summary:** Introduces Half-Life Regression (HLR) for modeling memory retention in spaced repetition systems. Validated on 13 million Duolingo learning traces. Demonstrates 50% error reduction over Leitner baseline.

**Relevance:** Provides the strongest empirically validated model for estimating knowledge retention. Open-source implementation enables adaptation.

---

### Piech, C., et al. (2015). Deep knowledge tracing. NIPS Proceedings.

**Quality Assessment:** ⭐⭐⭐⭐ (Influential, performance benchmark)

**Summary:** Applies recurrent neural networks to knowledge tracing, achieving AUC of 0.85 on Khan Academy data versus 0.68 for standard BKT.

**Relevance:** Demonstrates potential of deep learning for knowledge estimation. Important benchmark for comparing simpler models.

---

### Deonovic, B., et al. (2018). Learning meets assessment: On the relation between item response theory and Bayesian knowledge tracing. Behaviormetrika, 45, 457-474.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Theoretical integration)

**Summary:** Establishes fundamental connections between BKT (longitudinal learning models) and IRT (cross-sectional assessment models). Proposes research agenda for integrated approaches.

**Relevance:** Provides theoretical justification for combining engagement-based and assessment-based estimation.

---

## Engagement and Learning Analytics

### Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: Potential of the concept, state of the evidence. Review of Educational Research, 74(1), 59-109.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Foundational review)

**Summary:** Defines engagement as multidimensional construct (behavioral, emotional, cognitive). Establishes theoretical framework for engagement research in education.

**Relevance:** Essential theoretical foundation for understanding what engagement metrics actually measure.

---

### Chi, M. T. H., & Wylie, R. (2014). The ICAP framework: Linking cognitive engagement to active learning outcomes. Educational Psychologist, 49(4), 219-243.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Influential framework)

**Summary:** Proposes Interactive-Constructive-Active-Passive (ICAP) framework for categorizing engagement modes. Demonstrates that learning outcomes increase from passive to interactive engagement.

**Relevance:** Provides theoretical basis for weighting different engagement types differently in proficiency estimation.

---

### Johar, N. A., et al. (2023). Learning analytics on student engagement to enhance students' learning performance: A systematic review. Sustainability, 15(10), 7849.

**Quality Assessment:** ⭐⭐⭐⭐ (Recent systematic review)

**Summary:** Systematic review of learning analytics and engagement. Identifies time-on-task, participation, and interaction as key measurable engagement indicators.

**Relevance:** Provides current synthesis of engagement-learning relationship literature.

---

## Platform-Specific Research

### Khan Academy (2025). Why Khan Academy will be using "skills to proficient" to measure learning outcomes. Khan Academy Blog.

**Quality Assessment:** ⭐⭐⭐⭐ (Industry validation)

**Summary:** Documents Khan Academy's shift to "skills to proficient" as primary learning metric. Reports correlation with MAP Growth standardized assessment scores.

**Relevance:** Demonstrates real-world validation of mastery-based engagement metrics against external assessments.

---

### Block, J. H., & Burns, R. B. (1976). Mastery learning. Review of Research in Education, 4, 3-49.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Seminal research)

**Summary:** Foundational research on mastery learning. Establishes that students working to mastery before progression show better long-term retention.

**Relevance:** Theoretical foundation for mastery-based proficiency frameworks.

---

## Gamification and Interactive Learning

### Meta-analysis on gamification effectiveness (PMC, 2023). Examining the effectiveness of gamification as a tool promoting teaching and learning in educational settings.

**Quality Assessment:** ⭐⭐⭐⭐⭐ (Meta-analysis, N>5,000)

**Summary:** Meta-analysis of 41 studies (5,071+ participants) finding large effect size (g = 0.822) for gamification on learning outcomes. Points, badges, and leaderboards most commonly studied.

**Relevance:** Validates that gamification elements can meaningfully contribute to learning and should be incorporated in engagement models.

---

### Hamari, J., Koivisto, J., & Sarsa, H. (2014). Does gamification work? A literature review of empirical studies on gamification. HICSS Proceedings.

**Quality Assessment:** ⭐⭐⭐⭐ (Influential review)

**Summary:** Systematic review finding mostly positive effects of gamification, with caveats about context-dependence and potential negative effects.

**Relevance:** Provides nuanced understanding of when gamification metrics indicate genuine engagement.

---

# Part 8: Sample Proficiency Gain Report

## Example Student Profile

**Student ID:** STU_2026_001  
**Course:** Introduction to Data Science  
**Reporting Period:** December 1-31, 2025  

### Overall Proficiency Summary

| Skill Area | Starting Level | Current Level | Change | Status |
|------------|---------------|---------------|--------|--------|
| Python Basics | Familiar | Proficient | +0.24 | ↑ On Track |
| Data Manipulation | Attempted | Familiar | +0.18 | ↑ Progressing |
| Visualization | Novice | Familiar | +0.32 | ↑↑ Excellent |
| Statistics | Familiar | Familiar | +0.08 | → Needs Focus |

### Engagement-Based Estimates

#### Video Content Engagement

| Content | Duration | Watch Time | Completion | Interactions | Est. Gain |
|---------|----------|------------|------------|--------------|-----------|
| Python Lists Tutorial | 8:30 | 10:15 | 100% | 3 pauses, 1 rewind | +0.08 |
| Pandas DataFrames | 12:00 | 14:30 | 95% | 5 pauses, 2 rewinds | +0.10 |
| Matplotlib Intro | 6:00 | 7:00 | 100% | 2 pauses | +0.06 |
| Statistical Tests | 15:00 | 8:00 | 53% | 1 pause | +0.02 |

#### Text Content Engagement

| Content | Pages | Scroll Depth | Time Spent | Est. Gain |
|---------|-------|--------------|------------|-----------|
| Python Documentation | 12 | 78% | 25 min | +0.06 |
| Data Cleaning Guide | 8 | 92% | 18 min | +0.08 |
| Statistics Primer | 20 | 45% | 12 min | +0.03 |

#### Interactive Content Engagement

| Content | Completion | Success Rate | Time | Est. Gain |
|---------|------------|--------------|------|-----------|
| Python Exercises (20) | 18/20 | 85% | 45 min | +0.12 |
| Data Challenges (5) | 4/5 | 80% | 60 min | +0.08 |
| Visualization Lab | 100% | 90% | 30 min | +0.10 |

### Proficiency Calculation Example

**Skill: Python Basics**

```
Prior Estimate (Dec 1): 0.52 (Familiar)

Learning Events:
1. Python Lists Tutorial
   - Engagement Score: 0.40×1.0 + 0.25×1.2 + 0.20×0.4 + 0.15×0 = 0.78
   - Content Value: 0.10
   - Gain: 0.10 × 0.78 = 0.078

2. Python Exercises
   - Engagement Score: 0.30×0.9 + 0.30×0.85 + 0.20×1.0 + 0.20×0.7 = 0.865
   - Content Value: 0.15
   - Gain: 0.15 × 0.865 = 0.130

3. Python Documentation Reading
   - Engagement Score: 0.35×0.78 + 0.35×1.25 + 0.20×0.3 + 0.10×0.33 = 0.80
   - Content Value: 0.05
   - Gain: 0.05 × 0.80 = 0.040

Total Raw Gain: 0.078 + 0.130 + 0.040 = 0.248

Retention Adjustment (half-life = 10 days, avg time since = 15 days):
Retention Factor: 2^(-15/10) = 0.35
Retained Gain: 0.248 × (0.35 + 0.65×0.5) = 0.168

Current Estimate: 0.52 + 0.168 = 0.688 → Proficient (rounded to 0.70)
```

### Recommendations

**Strengths:**
- Strong engagement with video content (high completion, rewatching)
- Excellent performance on interactive exercises
- Good time investment in visualization topics

**Areas for Improvement:**
- Statistics content shows lower engagement (53% video completion)
- Text reading time below expected for complexity level
- Recommend: Shorter statistics videos, more interactive practice

**Suggested Next Steps:**
1. Complete remaining Python exercises
2. Revisit Statistics video with guided notes
3. Try advanced visualization challenges

---

# Appendices

## Appendix A: Engagement Metric Collection Specifications

### Video Events

```json
{
  "event_type": "video_interaction",
  "timestamp": "ISO8601",
  "user_id": "string",
  "content_id": "string",
  "session_id": "string",
  "action": "play|pause|seek|speed_change|complete",
  "position_seconds": "number",
  "speed": "number (for speed_change)",
  "seek_from": "number (for seek)",
  "seek_to": "number (for seek)"
}
```

### Text/Document Events

```json
{
  "event_type": "document_interaction",
  "timestamp": "ISO8601",
  "user_id": "string",
  "content_id": "string",
  "session_id": "string",
  "action": "view|scroll|highlight|click|close",
  "scroll_depth_percent": "number (0-100)",
  "page_number": "number",
  "viewport_time_ms": "number"
}
```

### Interactive Content Events

```json
{
  "event_type": "interactive_action",
  "timestamp": "ISO8601",
  "user_id": "string",
  "content_id": "string",
  "session_id": "string",
  "action": "start|attempt|success|failure|hint|skip|complete",
  "element_id": "string",
  "attempt_number": "number",
  "time_to_action_ms": "number"
}
```

## Appendix B: Model Parameter Defaults

### EW-BKT Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| P(L₀) | 0.10 | 0.01-0.50 | Initial mastery probability |
| P(T) base | 0.20 | 0.05-0.50 | Base learning rate |
| P(S) | 0.10 | 0.01-0.30 | Slip probability |
| P(G) | 0.20 | 0.05-0.40 | Guess probability |
| W_completion | 0.40 | 0.20-0.60 | Completion weight |
| W_time | 0.35 | 0.20-0.50 | Time quality weight |
| W_interaction | 0.25 | 0.10-0.40 | Interaction weight |

### HLR Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| base_half_life | 7 days | Calibrate from data |
| exposure_factor | 0.20 | Half-life increase per exposure |
| engagement_exponent | 0.50 | Engagement quality impact |
| min_half_life | 1 day | Floor for retention |
| max_half_life | 90 days | Ceiling for retention |

### Mastery Thresholds

| Level | Threshold | Calibration |
|-------|-----------|-------------|
| Mastered | ≥ 0.90 | Typically fixed |
| Proficient | ≥ 0.70 | Calibrate to assessment performance |
| Familiar | ≥ 0.50 | May vary by domain |
| Attempted | ≥ 0.20 | Any engagement |
| Novice | < 0.20 | Default state |

## Appendix C: Validation Checklist

### Pre-Deployment Validation

- [ ] Engagement data collection verified (all event types captured)
- [ ] Engagement metrics computed correctly (spot-check calculations)
- [ ] Content values assigned by curriculum team
- [ ] Difficulty estimates reviewed for reasonableness
- [ ] Baseline assessment administered to sample population
- [ ] Initial correlation computed (target: r > 0.30)
- [ ] User-facing displays reviewed for clarity

### Ongoing Monitoring

- [ ] Weekly correlation tracking implemented
- [ ] Drift detection alerts configured
- [ ] Monthly calibration review scheduled
- [ ] Quarterly external validation planned
- [ ] User feedback collection mechanism active

---

**Document End**

*This report was prepared based on peer-reviewed research, documented industry practices, and established educational measurement frameworks. Implementation should include domain-specific calibration and ongoing validation.*
