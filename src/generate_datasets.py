"""
Generate test datasets for prompt compression experiments.

Task 1 (ALC): Arbitrary Label Classification - 500 sentences across 3 sentiment categories
Task 2 (SLR): Synthetic Logic Rules - 600 inputs across 4 rule categories
"""

import json
import random
import os

random.seed(42)

# ──────────────────────────────────────────────
# Task 1: Arbitrary Label Classification (ALC)
# ──────────────────────────────────────────────

HAPPINESS_TEMPLATES = [
    "I just got {good_thing} and I'm thrilled!",
    "Today is the best day because {good_event}.",
    "I can't stop smiling after {good_event}.",
    "Nothing beats the feeling of {good_thing}!",
    "I'm so excited about {good_event}!",
    "What a wonderful {time_period}, everything is going great!",
    "I'm overjoyed that {good_event}.",
    "This is amazing, {good_event}!",
    "I feel fantastic after {good_event}.",
    "Life is beautiful when {good_event}.",
    "I'm beaming with pride because {good_event}.",
    "Incredible news — {good_event}!",
    "My heart is full after {good_event}.",
    "I woke up feeling absolutely {positive_adj} today.",
    "Everything came together perfectly and {good_event}.",
    "{good_event} and I couldn't be happier!",
    "Just found out about {good_thing} — this is the best!",
    "I'm dancing around the house because {good_event}.",
    "Pure bliss — {good_event}.",
    "What a {positive_adj} surprise, {good_event}!",
]

SADNESS_TEMPLATES = [
    "I'm really upset because {bad_event}.",
    "It breaks my heart that {bad_event}.",
    "I can't believe {bad_event}, I'm devastated.",
    "Today was terrible — {bad_event}.",
    "I feel so down after {bad_event}.",
    "Nothing seems to help, {bad_event}.",
    "I'm heartbroken over {bad_event}.",
    "This is so disappointing — {bad_event}.",
    "I've been crying because {bad_event}.",
    "I feel empty inside after {bad_event}.",
    "It's hard to go on when {bad_event}.",
    "I miss {lost_thing} so much it hurts.",
    "The worst part is that {bad_event}.",
    "I'm filled with sorrow because {bad_event}.",
    "My heart aches knowing that {bad_event}.",
    "I feel so lonely since {bad_event}.",
    "Everything reminds me of {lost_thing}.",
    "{bad_event} and I can't stop thinking about it.",
    "I wish {bad_event} never happened.",
    "It's a {negative_adj} day — {bad_event}.",
]

ANGER_TEMPLATES = [
    "I'm furious that {anger_event}!",
    "How dare they {anger_action}!",
    "I can't stand it when {anger_event}.",
    "This is absolutely unacceptable — {anger_event}!",
    "I'm so fed up with {anger_thing}.",
    "It infuriates me that {anger_event}.",
    "I've had it with {anger_thing}!",
    "What a ridiculous situation — {anger_event}.",
    "I'm livid because {anger_event}!",
    "They had the nerve to {anger_action}!",
    "I'm seething after {anger_event}.",
    "This makes my blood boil — {anger_event}.",
    "Unbelievable, {anger_event} again!",
    "I'm outraged that {anger_event}.",
    "Stop {anger_action}, it's driving me crazy!",
    "{anger_event} and nobody even cares!",
    "I want to scream because {anger_event}.",
    "How is it possible that {anger_event}?!",
    "I'm disgusted by {anger_thing}.",
    "Every time {anger_event}, I lose it.",
]

GOOD_THINGS = [
    "a promotion", "a new puppy", "concert tickets", "a scholarship",
    "flowers from a friend", "a surprise vacation", "a perfect score",
    "an award", "a raise", "a love letter", "great news from the doctor",
    "my dream job offer", "a birthday surprise", "a heartfelt compliment",
    "a reunion with old friends", "tickets to my favorite show",
]

GOOD_EVENTS = [
    "I got into my top choice university", "my best friend is visiting",
    "I finished the marathon", "the project was a huge success",
    "I learned to play my favorite song on guitar", "my team won the championship",
    "I passed all my exams", "my garden is blooming beautifully",
    "I reconnected with my childhood friend", "I got the apartment I wanted",
    "we adopted a rescue dog", "my book got published",
    "I surprised my parents with a trip", "my startup got funded",
    "I completed my first painting", "I ran my personal best time",
    "the kids threw me a surprise party", "I mastered a new recipe",
    "my proposal was accepted", "I finally beat my high score",
    "my volunteer work made a real difference", "I landed the lead role",
    "the sunset was absolutely gorgeous tonight", "I made a new friend at the park",
]

POSITIVE_ADJ = [
    "wonderful", "fantastic", "incredible", "amazing", "delightful",
    "magnificent", "spectacular", "glorious", "brilliant", "marvelous",
]

BAD_EVENTS = [
    "my pet ran away", "I failed the exam", "I lost my wallet",
    "my flight got cancelled", "I didn't get the job", "my phone screen cracked",
    "my best friend moved away", "I broke my favorite mug",
    "the concert was cancelled", "I got a parking ticket",
    "my project was rejected", "I missed an important deadline",
    "my car broke down on the highway", "I dropped my lunch on the floor",
    "I found out I wasn't invited", "they discontinued my favorite product",
    "my plant died", "I got stood up on a date",
    "my computer crashed and I lost my work", "I tore my favorite jacket",
    "the restaurant lost our reservation", "my bike was stolen",
    "I tripped in front of everyone", "rain ruined the outdoor wedding",
]

LOST_THINGS = [
    "my grandmother", "my old neighborhood", "simpler times",
    "my childhood home", "my dog", "those carefree summer days",
    "my late father", "the way things used to be", "my old friends",
    "my cat who passed", "the good old days",
]

NEGATIVE_ADJ = [
    "terrible", "awful", "dreadful", "miserable", "gloomy",
    "painful", "heartbreaking", "devastating", "depressing", "bleak",
]

ANGER_EVENTS = [
    "they cancelled my order without telling me",
    "the landlord raised rent again",
    "my coworker took credit for my work",
    "the meeting was a complete waste of time",
    "they changed the policy without notice",
    "the delivery was three weeks late",
    "they ignored my complaint",
    "the internet went down during my presentation",
    "someone cut in line right in front of me",
    "the store refused to honor the warranty",
    "they double-charged my credit card",
    "my neighbor plays loud music at 2am",
    "they lost my luggage for the third time",
    "the customer service put me on hold for an hour",
    "someone dented my car in the parking lot",
    "the new update broke everything",
    "they promised a refund but never sent it",
    "my package was marked delivered but never arrived",
    "they scheduled the meeting over my lunch break again",
    "the contractor did a terrible job and won't fix it",
]

ANGER_ACTIONS = [
    "lie to my face", "ignore my emails for a week",
    "cancel at the last minute", "blame me for their mistake",
    "talk behind my back", "show up an hour late with no apology",
    "take my parking spot", "eat my labeled food from the fridge",
    "break their promise", "interrupt me in the middle of speaking",
]

ANGER_THINGS = [
    "people who don't keep their word", "this broken system",
    "the constant delays", "being treated unfairly",
    "incompetent customer service", "this bureaucratic nonsense",
    "rude drivers on the highway", "the endless paperwork",
    "people who litter everywhere", "this unreliable software",
]

TIME_PERIODS = ["day", "morning", "week", "afternoon", "evening", "weekend"]


def fill_happiness():
    t = random.choice(HAPPINESS_TEMPLATES)
    return t.format(
        good_thing=random.choice(GOOD_THINGS),
        good_event=random.choice(GOOD_EVENTS),
        time_period=random.choice(TIME_PERIODS),
        positive_adj=random.choice(POSITIVE_ADJ),
    )


def fill_sadness():
    t = random.choice(SADNESS_TEMPLATES)
    return t.format(
        bad_event=random.choice(BAD_EVENTS),
        lost_thing=random.choice(LOST_THINGS),
        negative_adj=random.choice(NEGATIVE_ADJ),
    )


def fill_anger():
    t = random.choice(ANGER_TEMPLATES)
    return t.format(
        anger_event=random.choice(ANGER_EVENTS),
        anger_action=random.choice(ANGER_ACTIONS),
        anger_thing=random.choice(ANGER_THINGS),
    )


def generate_alc_dataset(n_per_class=167):
    """Generate ALC dataset: ~500 examples (167 per class)."""
    examples = []

    for _ in range(n_per_class):
        examples.append({"text": fill_happiness(), "label": "zorp"})
    for _ in range(n_per_class):
        examples.append({"text": fill_sadness(), "label": "bleem"})
    for _ in range(n_per_class):
        examples.append({"text": fill_anger(), "label": "quiff"})

    # Pad to exactly 501 (167*3) is fine, or add one more to sadness for 500
    random.shuffle(examples)
    return examples[:500]


# ──────────────────────────────────────────────
# Task 2: Synthetic Logic Rules (SLR)
# ──────────────────────────────────────────────

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "black", "white", "brown", "silver", "golden", "teal", "crimson",
]

ANIMALS = [
    "dog", "cat", "elephant", "tiger", "rabbit", "eagle", "dolphin",
    "horse", "bear", "snake", "whale", "parrot", "fox", "wolf",
    "penguin", "lion", "deer", "owl", "shark", "turtle",
]

CITIES = [
    "Paris", "London", "Tokyo", "Berlin", "Rome", "Sydney", "Cairo",
    "Mumbai", "Toronto", "Moscow", "Bangkok", "Lima", "Seoul", "Vienna",
    "Prague", "Dublin", "Athens", "Lisbon", "Oslo", "Helsinki",
]

NEGATIVE_WORDS = [
    "terrible", "awful", "horrible", "dreadful", "ugly", "disgusting",
    "miserable", "grim", "depressing", "bleak", "ruined", "destroyed",
    "tragic", "devastating", "corrupt", "polluted", "dangerous", "toxic",
]

NEUTRAL_NOUNS = [
    "table", "book", "window", "chair", "lamp", "painting", "garden",
    "bridge", "tower", "mountain", "river", "forest", "ocean", "cloud",
    "street", "building", "market", "library", "museum", "station",
]

NEUTRAL_ADJ = [
    "large", "small", "old", "new", "bright", "quiet", "busy",
    "ancient", "modern", "simple", "famous", "hidden", "central",
]

SENTENCE_FRAMES_WITH_NUM = [
    "The {adj} {noun} has {num} of them.",
    "I saw {num} {adj} {noun} today.",
    "There are {num} {noun} near the {noun2}.",
    "About {num} {noun} were found by the {noun2}.",
    "The {noun} counted {num} items on the {noun2}.",
]

SENTENCE_FRAMES_NO_NUM = [
    "The {adj} {noun} sits by the {noun2}.",
    "A {adj} {noun} appeared near the {noun2}.",
    "The {noun} moved through the {adj} {noun2}.",
    "We spotted a {adj} {noun} near the {noun2}.",
    "The {adj} {noun} is next to the {noun2}.",
]


def make_slr_alpha(n=150):
    """Rule 1: color AND number > 5 → ALPHA"""
    examples = []
    for _ in range(n):
        color = random.choice(COLORS)
        num = random.randint(6, 99)
        frame = random.choice(SENTENCE_FRAMES_WITH_NUM)
        noun = random.choice(NEUTRAL_NOUNS)
        noun2 = random.choice(NEUTRAL_NOUNS)
        text = frame.format(adj=color, noun=noun, noun2=noun2, num=num)
        examples.append({"text": text, "label": "ALPHA"})
    return examples


def make_slr_beta(n=150):
    """Rule 2: animal AND no numbers → BETA"""
    examples = []
    for _ in range(n):
        animal = random.choice(ANIMALS)
        frame = random.choice(SENTENCE_FRAMES_NO_NUM)
        adj = random.choice(NEUTRAL_ADJ)
        noun2 = random.choice(NEUTRAL_NOUNS)
        text = frame.format(adj=adj, noun=animal, noun2=noun2)
        examples.append({"text": text, "label": "BETA"})
    return examples


def make_slr_gamma(n=150):
    """Rule 3: city AND negative word → GAMMA"""
    examples = []
    for _ in range(n):
        city = random.choice(CITIES)
        neg = random.choice(NEGATIVE_WORDS)
        frames = [
            f"The situation in {city} is {neg}.",
            f"{city} has become {neg} lately.",
            f"People say {city} is {neg} these days.",
            f"Visiting {city} was a {neg} experience.",
            f"The news from {city} is quite {neg}.",
            f"Life in {city} feels {neg} right now.",
        ]
        text = random.choice(frames)
        examples.append({"text": text, "label": "GAMMA"})
    return examples


def make_slr_delta(n=150):
    """Rule 4: none of the above → DELTA"""
    examples = []
    for _ in range(n):
        adj = random.choice(NEUTRAL_ADJ)
        noun = random.choice(NEUTRAL_NOUNS)
        noun2 = random.choice(NEUTRAL_NOUNS)
        frame = random.choice(SENTENCE_FRAMES_NO_NUM)
        text = frame.format(adj=adj, noun=noun, noun2=noun2)
        examples.append({"text": text, "label": "DELTA"})
    return examples


def generate_slr_dataset():
    """Generate SLR dataset: 600 examples (150 per class)."""
    examples = []
    examples.extend(make_slr_alpha(150))
    examples.extend(make_slr_beta(150))
    examples.extend(make_slr_gamma(150))
    examples.extend(make_slr_delta(150))
    random.shuffle(examples)
    return examples


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate ALC dataset
    alc_data = generate_alc_dataset()
    alc_path = os.path.join(data_dir, "alc_test_set.jsonl")
    with open(alc_path, "w") as f:
        for ex in alc_data:
            f.write(json.dumps(ex) + "\n")
    print(f"ALC dataset: {len(alc_data)} examples written to {alc_path}")

    # Verify ALC label distribution
    from collections import Counter
    alc_counts = Counter(ex["label"] for ex in alc_data)
    print(f"  Label distribution: {dict(alc_counts)}")

    # Generate SLR dataset
    slr_data = generate_slr_dataset()
    slr_path = os.path.join(data_dir, "slr_test_set.jsonl")
    with open(slr_path, "w") as f:
        for ex in slr_data:
            f.write(json.dumps(ex) + "\n")
    print(f"SLR dataset: {len(slr_data)} examples written to {slr_path}")

    slr_counts = Counter(ex["label"] for ex in slr_data)
    print(f"  Label distribution: {dict(slr_counts)}")

    # Spot-check: print 5 examples from each dataset
    print("\n--- ALC Spot Check ---")
    for ex in alc_data[:5]:
        print(f"  [{ex['label']}] {ex['text']}")

    print("\n--- SLR Spot Check ---")
    for ex in slr_data[:5]:
        print(f"  [{ex['label']}] {ex['text']}")


if __name__ == "__main__":
    main()
