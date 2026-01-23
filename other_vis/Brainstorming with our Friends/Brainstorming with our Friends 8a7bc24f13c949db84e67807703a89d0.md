# Brainstorming with our Friends

Event time: February 17, 2025

[https://miro.com/app/board/uXjVMNNk_Ww=/](https://miro.com/app/board/uXjVMNNk_Ww=/)



## Visualizations

- https://dl.acm.org/doi/abs/10.1145/3343413.3377960
- [A beginner’s guide to eye tracking for psycholinguistic studies of reading | Behavior Research Methods](https://link.springer.com/article/10.3758/s13428-024-02572-4)
- https://github.com/tmalsburg/scanpath
- https://github.com/aeye-lab/pymovements
- https://arxiv.org/pdf/2311.06095
- [pmc.ncbi.nlm.nih.gov/articles/PMC7140773/pdf/jemr-10-05-i.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC7140773/pdf/jemr-10-05-i.pdf)
- https://www.semanticscholar.org/reader/c621ee0d2faee8d02e7760d72a2e2db091a9e121
- https://arxiv.org/pdf/2311.06095
- https://ceur-ws.org/Vol-3777/short2.pdf
- [eyekit API documentation](https://jwcarr.github.io/eyekit/)
- Repeated reading paper (TODO add link)
    
    ![image.png](Brainstorming%20with%20our%20Friends/image.png)
    
- Goal Decoding
    
    ![image.png](Brainstorming%20with%20our%20Friends/image%201.png)
    

## 28/4

Goals: Develop and release visualziation tools for eye movements in reading , 
hopefully useful for researchers in a programatic way (batch data) or GUI format (?).

- Start with programatic and then wrap with a GUI and support batch data.
- Simple GUI is simple to do, something people will actually use - not so much.

Venue: CHI?

Check out in vizualization /CHI community on meaningful evaluation?

How to represent word properties?

- color gradient / font size / color intensity of fixation/word/bounding box of word
- if fixation duration and word freq both represented on fixation. -easy to compare and doesn’t require the text.

Push forward practical aspects of the project.

Who does what?

Important to add detailed tutorials to empowering people to use the tool.

What should be the input?

Discuss collaboration with SR?

Checkout [pymovements video replay](https://github.com/aeye-lab/pymovements-videoreplay/tree/pymovements-videoreplay) demo.

Do a visualiation session in the retreat [Graph_presentation - Google Slides](https://docs.google.com/presentation/d/1mo2OR3mN3ast6wQZ9sWbXoZu9c2QVM09f4ENHFWG1og/edit#slide=id.p28)

Shubi: Uploade vis to repo

### Visualization Meeting Summary 14/4

- Next Meeting 28.4 10 am (same time).
- Github - https://github.com/theDebbister/visualization
- Properties:
    - Words:
        - Position on screen
        - Word order
        - Linguistic properties (text / length / freq / …)
    - Fixations
        - Position on screen (XY coordinates)
        - Position on words
        - Scanpath order (order of fixations)
        - Fixation duration
    - Saccades
    - Word-level Eye movement features
        - Other aggregated word-level measures
    - Time
- Metadata / properties of participants /  labels
- Raw data
    - scattering of a single fixation
    - Blinks
- Filtering:
    - Saccade type (regression / forward / etc)
    - N-pass (first / second / etc)
    - Text-based (critical span / words-of interest)
- Aggregation over multiple trials (text or participants)
- Comparison (overlaying two trials)
- Nuances:
    - Discrete / vs continuous representation
    - Lower quality eye trackers - approximate position and duration
    - Eye movements can be over sentences / paragraphs / multiple pages / code / image+text
    - Colors and shapes, and font sizes, intensity / shade
    - Interactive visualization builder - streamlit?
    - Think of elements as layers that can be optionally added (e.g. text, saccades, etc)
    - Outliers and noise
    - machine learning models can support higher dimensional input (e.g. layers can be different word-level features in heatmap view) or multiple inputs
    - Multiple eyes (2) (currently usually two colors / show just one / average)
    - Calibration quality
    - Input data (raw/fixation/wordlevel / specific info)
    - small multiples
- Visualization options:
    - 2D plots (preferred)
    - Graphs
    - 3D plots
    - Circular plots
    - Video over time
    
    ![image.png](Brainstorming%20with%20our%20Friends/image%202.png)
    
    ![image.png](Brainstorming%20with%20our%20Friends/image%203.png)
    

![image.png](Brainstorming%20with%20our%20Friends/image%204.png)

![image.png](Brainstorming%20with%20our%20Friends/image%205.png)