# Example evaluation metrics data

EXAMPLE_METRICS = {
    "Hallucination": {
        "prompt": """You are assessing a chat bot response to a user's input based on whether it contradicts the known ground truth. Focus on factual inconsistencies and opposing statements.\n
Score:
A score of 0 means the response contains no contradictions to the ground truth.
A score of 1 means the response contains contradictions to the ground truth.\n
Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Ground Truth Response]: {{ground_truth}}
***
[Response]: {{response}}
***
[END DATA]""",
        "input": """Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
A) Carbon dioxide 
B) Food
C) Protection
D) Water""",
        # Example from ARC Evals
        "ground_truth": "B) Food",
        "response": "C) Protection",
    },
    "Precision": {
        "prompt": """You are assessing a chat bot response to a user's input based on how completely it captures the key facts and details from the ground truth response.\n
Score:
A score of 1 means the response is completely irrelevant or contradicts the ground truth response.
A score of 2 means the response includes some relevant information but is mostly filled with irrelevant or incorrect details.
A score of 3 means the response is somewhat relevant with a balanced mix of relevant and irrelevant information.
A score of 4 means the response is mostly relevant, containing only minor irrelevant details.
A score of 5 means the response is entirely relevant, perfectly aligning with the ground truth response without any irrelevant information.\n
Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Ground Truth Response]: {{ground_truth}}
***
[Response]: {{response}}
***
[END DATA]""",
        # Example from Prometheus
        "input": "In a city, there are three different colored houses - red, blue, and yellow. The city council has made a rule that no two same colored houses can be adjacent to each other. A query is raised by the city planner who is designing the layout of the houses. The query is, 'Can you help me design a layout where I can place these houses without violating the rule of the city council?' The user didn't provide information about the total number of houses, the layout structure, or the proportion of house colors.",
        "ground_truth": "In order to provide an accurate layout design, I'll need a bit more information. Could you please specify how many houses in total are to be placed in the city? Also, it would be helpful to know the ratio of the different colored houses. Are there equal numbers of red, blue, and yellow houses, or does one color predominate? Furthermore, understanding the layout structure you have in mind is also important. Is it a linear layout, a circular one, a block-based one, or some other structure? Once I have these details, I'll be able to provide a more precise and useful answer.",
        "response": "In designing the layout, you could alternate between the three colors - red, blue, and yellow, to ensure no two houses of the same color are next to each other. You could possibly start with a red house, followed by a blue, then a yellow, and then repeat the sequence until all houses are placed. However, the total number of houses and the ratio of the different colored houses could affect the final layout. For instance, if there are more red houses than blue and yellow, some adjustments may need to be made. Also, the structure of the layout could play a role, whether it's a linear or block-based layout, or any other. It might be helpful to get some additional information about these aspects, but based on the information provided, this approach could work.",
    },
    "Recall": {
        "prompt": """You are assessing a chat bot response to a user's input based on how completely it captures the key facts and details from the ground truth response.\n
Score:
A score of 1 means the response misses almost all the key facts and details from the ground truth.
A score of 2 means the response includes only a few of the key facts and details from the ground truth.
A score of 3 means the response captures a moderate amount of the key facts and details from the ground truth.
A score of 4 means the response includes most of the key facts and details from the ground truth.
A score of 5 means the response perfectly captures all key facts and details from the ground truth without omission.\n
Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Ground Truth Response]: {{ground_truth}}
***
[Response]: {{response}}
***
[END DATA]""",
        # Example from Prometheus
        "input": """There's a thing that's bothering me. It's related to a kind of fruit, but not really the usual ones we eat. It's sweet, crunchy, and a lot of people like it, but it's not something you'd normally find in a grocery store. What kind of fruit am I talking about?""",
        "ground_truth": """Based on your description, it seems like you might be referring to the 'honeycrisp apple'. While it is a type of fruit and is known for its sweetness and crunchiness, it's not as commonly found in grocery stores as other fruit varieties due to its specific growing requirements and seasonality. Its unique taste and texture make it a favorite among many people. However, it's important to note that there could potentially be other fruits that also fit the given description, as taste and texture can vary significantly among fruit varieties.""",
        "response": """From your clues, it seems that you might be referring to a 'honeycrisp apple'. This fruit is known for its sweetness and crunchiness, but it isn't typically seen in many grocery stores. Many people do enjoy its unique taste and texture. However, without more specific details, there could be other fruits that also match your description. Some fruits can share similar characteristics like sweetness and crunchiness.""",
    },
    "Logical_Coherence": {
        "prompt": """You are assessing a chat bot response to a user's input based on the logical coherence of the response.\n
Score:
A score of 1 means the response is illogical, contains internal contradictions, or has significant jumps in reasoning that make it hard to follow.
A score of 3 means the response follows some logical flow but has minor inconsistencies or gaps in reasoning.
A score of 5 means the response is completely logically coherent, with clear connections between ideas and a natural progression of thought.\n
Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Response]: {{response}}
***
[END DATA]""",
        # Example from RewardBench
        "input": "How did US states get their names?",
        "response": "US states got their names for a variety of reasons. Many are named after early explorers, leaders, or patriots. Some are named after geographical features, such as mountains or rivers. Others are named after plants or animals that are native to the state. Some are even named after historical figures or characters from literature or mythology.",
    },
    "Faithfulness": {
        "prompt": """You are assessing a chat bot response to a user's input based on the faithfulness of the response to the context retrieved.\n
Score:
A score of 1 means that the response contains statements that directly contradict the context or are entirely unsupported by it.
A score of 3 means that the response contains some statements that are not directly supported by the context.
A score of 5 means that the response is fully grounded in the context, with all statements accurately reflecting the provided information.\n
Here is the data:
[BEGIN DATA]
***
[User Query]: {{input}}
***
[Retrieved Context]: {{context}}
*** 
[Response]: {{response}}
***
[END DATA]""",
        # Example from RAGTruth
        "input": "How to prepare to get an ultrasound?",
        "context": """Passage 1: 1 If you are just interested in checking the gender, you can go as early as 17 weeks. 2 Any earlier, and the results may not be accurate, leading to an extra coat of paint for your bedroom. 3 To get good pictures, 3D ultrasounds are best performed between 20 and 32 weeks, and the best pictures are between 24 and 30 weeks.""",
        "response": """ Based on the given passages, here’s how to prepare for an ultrasound: For a gender reveal ultrasound, wait until at least 17 weeks of pregnancy to ensure accurate results. For 3D ultrasounds, schedule the appointment between 20 and 32 weeks of pregnancy for the best pictures""",
    },
}
