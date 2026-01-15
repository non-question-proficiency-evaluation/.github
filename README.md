<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Non-Question Proficiency Evaluation</div>

## 1. Problem Statement

This organization addresses the core challenge of **estimating learner proficiency from non-assessment engagement data**. Traditional proficiency evaluation relies on direct assessment through questions and exercises. This project explores whether we can infer learner proficiency from indirect signals such as:

- Video watching behaviors (pauses, rewinds, playback speed)
- Reading interactions (scroll depth, time spent, reading speed)
- Interactive content engagement (action sequences, completion patterns)

The fundamental question is: **Can engagement patterns serve as valid proxies for learning proficiency when direct assessment data is unavailable?**

## 2. Approach

This project treats proficiency estimation as a **prediction problem** using machine learning models trained on engagement signals. Each content type requires specialized modeling approaches that capture the unique behavioral patterns associated with effective learning:

- **Video content** requires models that interpret cognitive load indicators (pauses, rewinds) as signals of productive learning
- **Reading content** needs models that track knowledge acquisition and decay from reading behaviors
- **Interactive content** benefits from sequence-based models that infer proficiency from action patterns

All models produce proficiency estimates that should be treated as **lower-confidence evidence** until validated by direct assessment.

## 3. Implementations

Three specialized implementations have been developed, each targeting a specific content type:

### 3.1. Interactive Content - AKT

**Directory**: `Interactive-Proficiency-AKT`

Uses **Context-Aware Attentive Knowledge Tracing (AKT)** to model learner proficiency from interactive engagement sequences. AKT employs attention mechanisms to capture contextual relationships between learning interactions and knowledge state transitions.

**Key Features**:
- Context-aware attention for modeling exercise relationships
- Supports both Rasch and Non-Rasch model variants
- Processes interaction sequences to predict future performance

### 3.2. Reading Content - LBKT

**Directory**: `Reading-Proficiency-LBKT`

Implements **Learning Behavior Knowledge Tracing (LBKT)** to track knowledge states from reading behaviors. The model captures multiple learning behaviors (Speed, Attempts, Hints) and their complex effects on learning and forgetting processes.

**Key Features**:
- Differentiated Behavior Effect Quantifying module
- Fused Behavior Effect Measuring module
- Forget gate mechanism for knowledge decay modeling

### 3.3. Video Content - SAINT+

**Directory**: `Video-Proficiency-SAINT`

Employs **SAINT+**, a Transformer-based knowledge tracing model adapted for video engagement data. The model uses encoder-decoder architecture with self-attention to process video interaction sequences and predict proficiency.

**Key Features**:
- Transformer-based architecture with encoder-decoder structure
- Temporal feature integration (time gaps, duration)
- Causal masking to prevent future data leakage
- Cross-validation AUC: 0.799

## 4. Directory Structure

```
Organization-non-question-proficiency-evaluation/
├── Interactive-Proficiency-AKT/    # AKT implementation for interactive content
├── Reading-Proficiency-LBKT/        # LBKT implementation for reading content
└── Video-Proficiency-SAINT/         # SAINT+ implementation for video content
```

Each directory contains:
- Model implementation code
- Data processing utilities
- Training and evaluation scripts
- Detailed README with usage instructions

## 5. Technical Considerations

- **Uncertainty**: Proficiency estimates from engagement data are probabilistic and should be validated against direct assessment
- **Content-Specific Modeling**: Each content type requires specialized approaches due to different behavioral signal characteristics
- **Temporal Dynamics**: Models account for knowledge acquisition, retention, and decay over time
- **Sequence Processing**: All implementations process engagement as temporal sequences to capture learning progression
