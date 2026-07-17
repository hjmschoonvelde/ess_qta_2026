# QTA Lab 10: LLM-Assisted Coding


## Learning Goals

In this lab, you will learn how to:

- define a qualitative coding task that can be applied at scale;
- write a transparent `quallmer` codebook;
- use `ellmer` syntax for model/provider settings;
- code leader speeches for credit claiming and blame attribution;
- compare coded claims across Labour and Conservative leaders;
- validate LLM-coded output against human judgements;
- optionally set up Ollama for local model use.

The substantive question is:

> Which Labour and Conservative leaders are more likely to claim credit,
> and which are more likely to attribute blame?

This is a useful task for LLM-assisted coding because the categories are
interpretive. We are not only asking whether a speech contains words
like “success” or “failure”. We are asking whether the speaker uses the
text to claim responsibility for a positive outcome, attribute
responsibility for a negative outcome, both, or neither.

`quallmer` is by no means the only way to interact with LLMs from R. You
can also use `ellmer` directly, provider-specific packages, web
interfaces, or local tools. We use `quallmer` here because it encourages
a structured qualitative coding workflow: define a codebook, request
structured outputs, compare coding runs, validate against human
judgements, and keep an audit trail.

## Start Here: Install And Test Ollama

The easiest way to try LLM-assisted coding without an API key is to run
a small open-weight model locally with Ollama. This follows the setup
used in the `quallmer` Ollama tutorial.

First, install the Ollama app from <https://ollama.com/download>. Start
the app outside R. On Windows or macOS, you can usually do this by
clicking the Ollama app icon. On macOS with Homebrew, you can also
install and start it from the terminal:

``` bash
brew install ollama
ollama serve
```

Then install the R package `rollama`, check that R can see Ollama, and
download a small teaching model. The model `llama3.2:1b` is relatively
small, so it is a good starting point for class. It is not necessarily
the best model for research.

``` r
install.packages("rollama")

library(rollama)

ping_ollama()

pull_model("llama3.2:1b")
```

Test the model from R:

``` r
rollama::query(
  "In one sentence, explain the difference between claiming credit and attributing blame.",
  model = "llama3.2:1b"
)
```

Check which models are installed using `rollama`:

``` r
if (requireNamespace("rollama", quietly = TRUE)) {
  rollama::list_models()
} else {
  message("Install rollama first with install.packages(\"rollama\").")
}
```

    # A tibble: 9 × 16
      name      model modified_at    size digest parent_model format family families
      <chr>     <chr> <chr>         <dbl> <chr>  <chr>        <chr>  <chr>  <list>  
    1 llama3.2… llam… 2026-07-16… 1.32e 9 baf6a… ""           gguf   llama  <chr>   
    2 deepseek… deep… 2025-07-17… 5.23e 9 69958… ""           gguf   qwen3  <chr>   
    3 llama3.2… llam… 2025-07-12… 2.02e 9 a80c4… ""           gguf   llama  <chr>   
    4 llama3.1… llam… 2024-07-24… 4.66e 9 a3403… ""           gguf   llama  <chr>   
    5 llama2:l… llam… 2024-07-18… 3.83e 9 78e26… ""           gguf   llama  <chr>   
    6 llava:la… llav… 2024-02-23… 4.73e 9 8dd30… ""           gguf   llama  <chr>   
    7 llava:la… llav… 2024-02-23… 4.73e 9 8dd30… ""           gguf   llama  <chr>   
    8 mixtral:… mixt… 2024-02-23… 2.64e10 7708c… ""           gguf   llama  <chr>   
    9 vicuna:l… vicu… 2024-02-23… 3.83e 9 37073… ""           gguf   llama  <NULL>  
    # ℹ 7 more variables: parameter_size <chr>, quantization_level <chr>,
    #   context_length <int>, embedding_length <int>, `1` <chr>, `2` <chr>,
    #   capabilities <chr>

``` r
rollama::list_models()
```

    # A tibble: 9 × 16
      name      model modified_at    size digest parent_model format family families
      <chr>     <chr> <chr>         <dbl> <chr>  <chr>        <chr>  <chr>  <list>  
    1 llama3.2… llam… 2026-07-16… 1.32e 9 baf6a… ""           gguf   llama  <chr>   
    2 deepseek… deep… 2025-07-17… 5.23e 9 69958… ""           gguf   qwen3  <chr>   
    3 llama3.2… llam… 2025-07-12… 2.02e 9 a80c4… ""           gguf   llama  <chr>   
    4 llama3.1… llam… 2024-07-24… 4.66e 9 a3403… ""           gguf   llama  <chr>   
    5 llama2:l… llam… 2024-07-18… 3.83e 9 78e26… ""           gguf   llama  <chr>   
    6 llava:la… llav… 2024-02-23… 4.73e 9 8dd30… ""           gguf   llama  <chr>   
    7 llava:la… llav… 2024-02-23… 4.73e 9 8dd30… ""           gguf   llama  <chr>   
    8 mixtral:… mixt… 2024-02-23… 2.64e10 7708c… ""           gguf   llama  <chr>   
    9 vicuna:l… vicu… 2024-02-23… 3.83e 9 37073… ""           gguf   llama  <NULL>  
    # ℹ 7 more variables: parameter_size <chr>, quantization_level <chr>,
    #   context_length <int>, embedding_length <int>, `1` <chr>, `2` <chr>,
    #   capabilities <chr>

You can also check the same thing from the terminal:

``` bash
ollama list
```

Once a model is available, `quallmer` can use it through the model
string `"ollama/model_name"`:

``` r
model_name <- "ollama/llama3.2:1b"
```

Later in the lab, we will also show the lower-level `ellmer` syntax:

``` r
local_chat <- ellmer::chat_ollama(
  model = "llama3.2:1b",
  params = ellmer::params(temperature = 0)
)
```

## Package Setup

`quallmer` organises the coding workflow: codebooks, coding runs,
validation, comparison, and provenance. `ellmer` provides the model
interface underneath. Live LLM calls require a working provider setup
and usually an API key or a local Ollama model. In this lab, code that
would contact an LLM is shown for you to inspect and run manually, but
Quarto will skip it when rendering the document. This keeps the lab
renderable even if you have not set up an API key or local model yet.

``` r
quallmer_installed <- requireNamespace("quallmer", quietly = TRUE)
ellmer_installed <- requireNamespace("ellmer", quietly = TRUE)

tibble::tibble(
  package = c("quallmer", "ellmer"),
  available = c(quallmer_installed, ellmer_installed)
)
```

    # A tibble: 2 × 2
      package  available
      <chr>    <lgl>    
    1 quallmer TRUE     
    2 ellmer   TRUE     

If needed, install or update the packages before running live LLM calls.

``` r
install.packages("ellmer")

# If quallmer is not yet available from your usual repository:
# install.packages("pak")
# pak::pak("SeraphineM/quallmer")
```

``` r
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(ggplot2)
library(quallmer)
library(ellmer)

party_colors <- c(
  "Conservative" = "#003B73",
  "Labour" = "#B00020"
)
```

## Read The Leader Corpus

We use the same House of Commons leader corpus as Labs 6 and 7. The
corpus contains up to 200 speeches by each Labour and Conservative party
leader in each legislative period from the 47th Parliament, beginning
with the 3 May 1979 general election, to the 58th Parliament, beginning
with the 4 July 2024 general election.

Unlike Lab 6, we do not aggregate speeches into leader-period documents.
Here the unit of analysis is one speech. That is important: credit
claiming and blame attribution often occur in particular interventions,
questions, or debate contributions.

``` r
data_path <- "Data/hc_leader_period_sample_1979_2024.rds"

hoc_leaders <- readRDS(data_path) |>
  filter(party %in% c("Conservative", "Labour")) |>
  mutate(
    year = as.integer(format(date, "%Y")),
    speech_id = paste0("leader_", row_number()),
    party = factor(party, levels = c("Conservative", "Labour")),
    leader_period = paste(leader, election_period, sep = ": ")
  )

hoc_leaders |>
  count(party, leader, sort = TRUE)
```

    # A tibble: 21 × 3
       party        leader                n
       <fct>        <chr>             <int>
     1 Labour       Tony Blair          800
     2 Conservative David Cameron       600
     3 Conservative Margaret Thatcher   600
     4 Labour       Jeremy Corbyn       495
     5 Conservative John Major          425
     6 Conservative Boris Johnson       400
     7 Conservative Theresa May         400
     8 Labour       Keir Starmer        400
     9 Labour       Neil Kinnock        348
    10 Conservative Rishi Sunak         248
    # ℹ 11 more rows

For LLM-assisted coding, start small. We create a balanced teaching
sample across leaders. In a research project, we would code more
speeches and validate the output carefully.

``` r
llm_pool <- hoc_leaders |>
  filter(
    terms >= 80,
    !is.na(text),
    text != ""
  )

set.seed(20260710)

leader_llm <- llm_pool |>
  group_by(party, leader) |>
  mutate(random_order = runif(n())) |>
  arrange(random_order, .by_group = TRUE) |>
  slice_head(n = 4) |>
  ungroup() |>
  select(-random_order) |>
  mutate(
    text_for_coding = str_squish(text),
    text_for_coding = str_trunc(text_for_coding, width = 1800, side = "right")
  )

leader_llm |>
  select(speech_id, date, party, leader, election_period, agenda, text_for_coding) |>
  slice_head(n = 4)
```

    # A tibble: 4 × 7
      speech_id   date       party     leader election_period agenda text_for_coding
      <chr>       <date>     <fct>     <chr>  <chr>           <chr>  <chr>          
    1 leader_4898 2019-09-25 Conserva… Boris… 56th Parliamen… Prime… I repeat that …
    2 leader_5492 2021-01-06 Conserva… Boris… 57th Parliamen… Covid… I thank my hon…
    3 leader_4975 2019-10-23 Conserva… Boris… 56th Parliamen… Engag… My right hon. …
    4 leader_5518 2021-04-28 Conserva… Boris… 57th Parliamen… Engag… I know that th…

The `text_for_coding` column is truncated because many models have
context limits, and because short examples are easier to inspect. In a
real project, truncation is a measurement decision. You would need to
report it and test whether it changes the coding.

## The Coding Task

We will code whether each speech contains:

- a **credit claim**: the speaker claims that they, their party, their
  government, or a relevant actor delivered a positive outcome;
- a **blame attribution**: the speaker attributes a negative outcome,
  failure, harm, or problem to another actor;
- **both**: the speaker contrasts their own positive record with another
  actor’s failure;
- **neither**: the speech does not contain a clear credit or blame
  claim.

This task is more interpretive than a dictionary count. A speech can
mention “jobs”, “waiting lists”, or “inflation” without claiming credit
or assigning blame. We are interested in the rhetorical move: who is
made responsible for what?

## Codebook Categories

``` r
claim_type_labels <- c(
  "credit_claim",
  "blame_attribution",
  "mixed_credit_and_blame",
  "neither"
)

target_actor_labels <- c(
  "own_party_or_government",
  "opposing_party_or_government",
  "public_body",
  "private_actor",
  "international_actor",
  "unclear_or_none"
)

policy_area_labels <- c(
  "economy",
  "public_services",
  "immigration",
  "foreign_affairs",
  "environment_energy",
  "constitutional_procedural",
  "other_or_unclear"
)
```

These labels are deliberately compact. If the codebook has too many
categories, both human coders and LLMs will struggle to apply it
consistently. If it has too few, we may lose the qualitative detail that
made LLM-assisted coding useful.

## Structured Output Schema

The codebook tells the model how to think about the concept. The schema
tells the model what the answer must look like. In the block below:

- `type_object()` says that each coded speech should be returned as one
  structured record;
- `type_boolean()` creates TRUE/FALSE fields for credit claiming and
  blame attribution;
- `type_enum()` restricts the model to predefined labels;
- `type_string()` allows short open-text answers where we want
  qualitative detail;
- `type_number()` asks for a numeric confidence score between 0 and 1.

This matters because we want output that can be inspected like
qualitative coding, but also analysed like a data frame.

``` r
credit_blame_schema <- ellmer::type_object(
  credit_claim = ellmer::type_boolean(
    "TRUE if the speaker claims credit for a positive outcome, improvement, achievement, or delivery."
  ),
  blame_attribution = ellmer::type_boolean(
    "TRUE if the speaker attributes a negative outcome, failure, harm, or problem to an actor."
  ),
  main_claim_type = ellmer::type_enum(
    claim_type_labels,
    "Overall classification of the speech."
  ),
  target_actor = ellmer::type_enum(
    target_actor_labels,
    "Actor mainly credited or blamed."
  ),
  target_name = ellmer::type_string(
    "Specific actor credited or blamed, if stated. Use 'unclear' if not stated."
  ),
  policy_area = ellmer::type_enum(
    policy_area_labels,
    "Main policy area of the credit or blame claim."
  ),
  confidence = ellmer::type_number(
    "Confidence in the coding decision from 0 to 1."
  ),
  evidence = ellmer::type_string(
    "Short quote or close paraphrase supporting the coding decision."
  )
)

credit_blame_schema
```

    <ellmer::TypeObject>
     @ description          : NULL
     @ required             : logi TRUE
     @ properties           :List of 8
     .. $ credit_claim     : <ellmer::TypeBasic>
     ..  ..@ description: chr "TRUE if the speaker claims credit for a positive outcome, improvement, achievement, or delivery."
     ..  ..@ required   : logi TRUE
     ..  ..@ type       : chr "boolean"
     .. $ blame_attribution: <ellmer::TypeBasic>
     ..  ..@ description: chr "TRUE if the speaker attributes a negative outcome, failure, harm, or problem to an actor."
     ..  ..@ required   : logi TRUE
     ..  ..@ type       : chr "boolean"
     .. $ main_claim_type  : <ellmer::TypeEnum>
     ..  ..@ description: chr "Overall classification of the speech."
     ..  ..@ required   : logi TRUE
     ..  ..@ values     : chr [1:4] "credit_claim" "blame_attribution" "mixed_credit_and_blame" "neither"
     .. $ target_actor     : <ellmer::TypeEnum>
     ..  ..@ description: chr "Actor mainly credited or blamed."
     ..  ..@ required   : logi TRUE
     ..  ..@ values     : chr [1:6] "own_party_or_government" "opposing_party_or_government" "public_body" "private_actor" ...
     .. $ target_name      : <ellmer::TypeBasic>
     ..  ..@ description: chr "Specific actor credited or blamed, if stated. Use 'unclear' if not stated."
     ..  ..@ required   : logi TRUE
     ..  ..@ type       : chr "string"
     .. $ policy_area      : <ellmer::TypeEnum>
     ..  ..@ description: chr "Main policy area of the credit or blame claim."
     ..  ..@ required   : logi TRUE
     ..  ..@ values     : chr [1:7] "economy" "public_services" "immigration" "foreign_affairs" ...
     .. $ confidence       : <ellmer::TypeBasic>
     ..  ..@ description: chr "Confidence in the coding decision from 0 to 1."
     ..  ..@ required   : logi TRUE
     ..  ..@ type       : chr "number"
     .. $ evidence         : <ellmer::TypeBasic>
     ..  ..@ description: chr "Short quote or close paraphrase supporting the coding decision."
     ..  ..@ required   : logi TRUE
     ..  ..@ type       : chr "string"
     @ additional_properties: logi FALSE

## Define The quallmer Codebook

The codebook is the measurement instrument. It should contain enough
information for someone else to understand what the model was asked to
do.

The `qlm_codebook()` object has four main parts:

- `name`: a short label for the coding task. This is useful when saving,
  comparing, or reporting coding runs.
- `role`: the perspective we ask the model to adopt. Here, we ask it to
  act like a careful qualitative coder of parliamentary speeches, not
  like a political commentator or general chatbot.
- `instructions`: the substantive coding rules. These define credit
  claiming, blame attribution, mixed cases, and cases that should be
  coded as neither. They also tell the model not to use outside
  knowledge about parties or leaders.
- `schema`: the structured output format we defined above. The
  instructions tell the model how to decide; the schema tells it how to
  return the decision.

Think of the codebook as the equivalent of a human-coder instruction
sheet. If the codebook is vague, the model may still return confident
answers, but those answers will be harder to interpret and validate.

``` r
credit_blame_codebook <- qlm_codebook(
  name = "HoC leader credit claiming and blame attribution",
  role = "You are a careful qualitative coder of UK parliamentary speeches.",
  instructions = paste(
    "Code the speech for credit claiming and blame attribution.",
    "Use only the information in the speech text.",
    "A credit claim says or implies that the speaker, their party, their government, or another actor delivered a positive outcome, improvement, achievement, or successful policy.",
    "A blame attribution says or implies that an actor caused, worsened, failed to address, or is responsible for a negative outcome, failure, harm, or public problem.",
    "Code mixed_credit_and_blame when the speech clearly contains both credit claiming and blame attribution.",
    "Code neither when the speech is descriptive, procedural, or does not clearly claim credit or attribute blame.",
    "Do not infer from party identity, leader identity, or outside knowledge.",
    "If several claims appear, code the most substantively developed claim.",
    "Evidence should be short and should justify both the main_claim_type and the target_actor."
  ),
  schema = credit_blame_schema
)

credit_blame_codebook
```

    quallmer codebook: HoC leader credit claiming and blame attribution 
      Input type:   text
      Role:         You are a careful qualitative coder of UK parliamentary spee...
      Instructions: Code the speech for credit claiming and blame attribution. U...
      Output schema:ellmer::TypeObject

## A Concrete Example

The next block uses a fixed example from the leader corpus. This is
better than drawing the example from the random teaching sample, because
the human coding should clearly match the text students see.

``` r
example_speech <- hoc_leaders |>
  filter(
    leader == "David Cameron",
    date == as.Date("2011-11-15"),
    str_detect(str_to_lower(text), fixed("we have delivered more apprenticeships"))
  ) |>
  slice_head(n = 1) |>
  mutate(text_for_coding = str_trunc(str_squish(text), width = 900))

example_speech |>
  select(date, party, leader, agenda, text_for_coding)
```

    # A tibble: 1 × 5
      date       party        leader        agenda                   text_for_coding
      <date>     <fct>        <chr>         <chr>                    <chr>          
    1 2011-11-15 Conservative David Cameron Topical Questions [Oral… The right hon.…

A plausible human coding might look like this:

``` r
tibble::tibble(
  credit_claim = TRUE,
  blame_attribution = TRUE,
  main_claim_type = "mixed_credit_and_blame",
  target_actor = "opposing_party_or_government",
  target_name = "Labour government",
  policy_area = "economy",
  confidence = 0.85,
  evidence = "contrasts Labour's NEET record with the claim that 'we have delivered more apprenticeships'"
)
```

    # A tibble: 1 × 8
      credit_claim blame_attribution main_claim_type        target_actor target_name
      <lgl>        <lgl>             <chr>                  <chr>        <chr>      
    1 TRUE         TRUE              mixed_credit_and_blame opposing_pa… Labour gov…
    # ℹ 3 more variables: policy_area <chr>, confidence <dbl>, evidence <chr>

The important point is that the evidence field should point back to the
text. Without evidence, it is difficult to tell whether the model made a
defensible coding decision.

## A First Ollama Check With ellmer

Before using `quallmer`, it is useful to see the lower-level `ellmer`
syntax. The code below creates a chat object for a local Ollama model
and sends one message to it.

The chunk is not run automatically when rendering the lab. Run it
manually once Ollama is installed, running, and the model has been
downloaded.

``` r
chat <- ellmer::chat_ollama(
  model = "llama3.2:1b",
  params = ellmer::params(temperature = 0)
)

chat$chat(
  "In one sentence, distinguish credit claiming from blame attribution."
)
```

## Code Speeches With Ollama And quallmer

Named character vectors are useful because `quallmer` stores names as
`.id` values in the coded output.

``` r
leader_texts <- leader_llm$text_for_coding
names(leader_texts) <- leader_llm$speech_id

head(names(leader_texts))
```

    [1] "leader_4898" "leader_5492" "leader_4975" "leader_5518" "leader_4350"
    [6] "leader_3861"

For classroom speed, we code a small balanced subset of speeches. The
model used below is local, but it still takes time to run. That is why
the live model call is shown but not run automatically during rendering.
Run this chunk manually before running the inspection, plotting,
validation, and provenance chunks below.

``` r
ollama_demo_ids <- c(
  "leader_4898", "leader_5492", "leader_4975", "leader_5518",
  "leader_3939", "leader_3968", "leader_3927", "leader_3916"
)

ollama_demo_metadata <- leader_llm |>
  filter(speech_id %in% ollama_demo_ids) |>
  mutate(order = match(speech_id, ollama_demo_ids)) |>
  arrange(order) |>
  select(-order)

ollama_demo_texts <- ollama_demo_metadata$text_for_coding
names(ollama_demo_texts) <- ollama_demo_metadata$speech_id

ollama_demo_metadata |>
  select(speech_id, date, party, leader, agenda)
```

    # A tibble: 8 × 5
      speech_id   date       party        leader        agenda                      
      <chr>       <date>     <fct>        <chr>         <chr>                       
    1 leader_4898 2019-09-25 Conservative Boris Johnson Prime Minister's Update     
    2 leader_5492 2021-01-06 Conservative Boris Johnson Covid-19                    
    3 leader_4975 2019-10-23 Conservative Boris Johnson Engagements [Oral Answers t…
    4 leader_5518 2021-04-28 Conservative Boris Johnson Engagements                 
    5 leader_3939 2011-07-13 Labour       Ed Miliband   Rupert Murdoch and News Cor…
    6 leader_3968 2012-02-22 Labour       Ed Miliband   Engagements [Oral Answers t…
    7 leader_3927 2011-05-03 Labour       Ed Miliband   <NA>                        
    8 leader_3916 2011-03-21 Labour       Ed Miliband   United Nations Security Cou…

The next chunk is the actual `quallmer` call that sends the selected
speeches to a local Ollama model for coding.

- `ollama_demo_texts` is the named vector of leader speeches.
- `credit_blame_codebook` contains the coding instructions and
  structured output schema.
- `model = "ollama/llama3.2"` tells `ellmer` to use the locally
  installed Ollama model `llama3.2`.
- `params = ellmer::params(temperature = 0)` makes the model output as
  stable as possible.
- `name` gives the coding run a clear label for provenance and later
  comparison.

``` r
coded_leaders_ollama <- qlm_code(
  ollama_demo_texts,
  credit_blame_codebook,
  model = "ollama/llama3.2",
  params = ellmer::params(temperature = 0),
  name = "leader_credit_blame_llama32_temperature0_demo"
)

coded_leaders_ollama
```

If you only have the smaller teaching model installed, change the model
string to `"ollama/llama3.2:1b"`. It will run faster, but the output may
be less reliable.

## Inspect Ollama Output

After running the live `qlm_code()` call, inspect the object returned by
`quallmer`. From this point onward, we use only one model-output object:
`coded_leaders_ollama`.

``` r
coded_leaders_ollama |>
  count(main_claim_type, sort = TRUE)
```

## Join Codes Back To Metadata

The model output only contains the coding results and the `.id` for each
speech. To ask substantive questions about parties and leaders, we need
to join those codes back to the original metadata: date, party, leader,
election period, agenda, and the speech text.

``` r
coded_with_metadata <- coded_leaders_ollama |>
  left_join(
    leader_llm |>
      select(
        speech_id,
        date,
        year,
        party,
        leader,
        election_period,
        agenda,
        text_for_coding
      ),
    by = c(".id" = "speech_id")
  )

coded_with_metadata |>
  select(
    .id,
    party,
    leader,
    election_period,
    main_claim_type,
    target_actor,
    policy_area,
    confidence
  ) |>
  slice_head(n = 10)
```

## Compare Parties And Leaders

The next table asks the substantive question directly: in this coded
sample, which leaders have higher estimated rates of credit claiming and
blame attribution?

``` r
leader_claim_rates <- coded_with_metadata |>
  group_by(party, leader) |>
  summarise(
    n = n(),
    credit_rate = mean(credit_claim),
    blame_rate = mean(blame_attribution),
    mixed_rate = mean(main_claim_type == "mixed_credit_and_blame"),
    .groups = "drop"
  ) |>
  arrange(party, desc(blame_rate), desc(credit_rate))

leader_claim_rates
```

``` r
claim_plot_data <- leader_claim_rates |>
  pivot_longer(
    cols = c(credit_rate, blame_rate),
    names_to = "claim_measure",
    values_to = "rate"
  ) |>
  mutate(
    claim_measure = factor(
      recode(
        claim_measure,
        credit_rate = "Credit claiming",
        blame_rate = "Blame attribution"
      ),
      levels = c("Credit claiming", "Blame attribution")
    ),
    leader_label = paste(leader, party, sep = "\n")
  )

leader_levels <- leader_claim_rates |>
  arrange(party, leader) |>
  mutate(leader_label = paste(leader, party, sep = "\n")) |>
  pull(leader_label)

claim_plot_data |>
  mutate(
    leader_label = factor(leader_label, levels = rev(leader_levels))
  ) |>
  ggplot(aes(x = claim_measure, y = leader_label)) +
  geom_tile(
    aes(fill = party),
    alpha = 0.28,
    color = "white",
    linewidth = 1.2,
    width = 0.95,
    height = 0.85
  ) +
  geom_text(
    aes(label = scales::percent(rate, accuracy = 1)),
    size = 5,
    fontface = "bold",
    color = "grey15"
  ) +
  scale_fill_manual(values = party_colors) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Party"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(lineheight = 0.95),
    legend.position = "bottom"
  )
```

This plot is based on a small Ollama-coded sample, so we should not
interpret the substantive pattern as evidence. The workflow is the key
point: code speeches with Ollama, join codes to metadata, then compare
leaders.

## Inspect The Evidence

The evidence field matters because it makes the coding easier to audit.
If the evidence does not support the label, the model may have guessed,
over-generalised, or relied on irrelevant cues.

``` r
coded_with_metadata |>
  filter(main_claim_type != "neither") |>
  select(
    date,
    party,
    leader,
    agenda,
    main_claim_type,
    target_actor,
    policy_area,
    evidence
  ) |>
  slice_head(n = 8)
```

## Check Output Consistency

Structured output makes it easier to check whether the model
contradicted itself. Here, `main_claim_type` should agree with the two
Boolean fields:

- `credit_claim` means credit is TRUE and blame is FALSE;
- `blame_attribution` means blame is TRUE and credit is FALSE;
- `mixed_credit_and_blame` means both are TRUE;
- `neither` means both are FALSE.

``` r
checked_ollama_output <- coded_leaders_ollama |>
  mutate(
    expected_main_claim_type = case_when(
      credit_claim & blame_attribution ~ "mixed_credit_and_blame",
      credit_claim ~ "credit_claim",
      blame_attribution ~ "blame_attribution",
      TRUE ~ "neither"
    ),
    internally_consistent = main_claim_type == expected_main_claim_type
  )

checked_ollama_output |>
  select(
    .id,
    credit_claim,
    blame_attribution,
    main_claim_type,
    expected_main_claim_type,
    internally_consistent
  )
```

If this check fails, do not simply fix the data silently. Inspect the
speech, revise the codebook if needed, and rerun the model. In a real
project, internal consistency is only the first check. We still need
human validation.

## Validate Against Human Coding

No LLM coding project is complete without validation. A practical
strategy is to human-code a stratified sample across parties, leaders,
and model-predicted labels. This matters here because credit and blame
may look different in government and opposition, and across leaders with
different rhetorical styles.

Here we create a small mock human-coded set. In your project, this
should come from actual human coding.

``` r
human_gold <- coded_leaders_ollama |>
  slice_head(n = 12) |>
  select(
    .id,
    credit_claim,
    blame_attribution,
    main_claim_type,
    target_actor,
    policy_area
  ) |>
  mutate(
    main_claim_type = case_when(
      row_number() %in% c(3, 8) ~ "neither",
      TRUE ~ main_claim_type
    ),
    credit_claim = main_claim_type %in% c("credit_claim", "mixed_credit_and_blame"),
    blame_attribution = main_claim_type %in% c("blame_attribution", "mixed_credit_and_blame")
  )

human_gold
```

With real `quallmer` objects, convert the human data and validate.

``` r
human_coded <- qlm_humancoded(
  human_gold,
  name = "expert_human_coder",
  codebook = credit_blame_codebook,
  texts = leader_texts,
  metadata = list(
    coder_training = "Instructor-coded validation sample",
    date = Sys.Date()
  )
)

qlm_validate(
  coded_leaders_ollama,
  human_coded,
  by = main_claim_type,
  level = "nominal",
  average = "none"
)
```

Once you have both model-coded and human-coded data, you can inspect
disagreements directly.

``` r
example_validation <- coded_leaders_ollama |>
  inner_join(
    human_gold |>
      rename(
        human_credit_claim = credit_claim,
        human_blame_attribution = blame_attribution,
        human_main_claim_type = main_claim_type,
        human_target_actor = target_actor,
        human_policy_area = policy_area
      ),
    by = ".id"
  ) |>
  mutate(
    claim_type_correct = main_claim_type == human_main_claim_type,
    credit_correct = credit_claim == human_credit_claim,
    blame_correct = blame_attribution == human_blame_attribution
  )

example_validation |>
  count(human_main_claim_type, main_claim_type)

example_validation |>
  summarise(
    claim_type_accuracy = mean(claim_type_correct),
    credit_accuracy = mean(credit_correct),
    blame_accuracy = mean(blame_correct),
    n = n()
  )
```

## Provenance And Reporting

`quallmer` keeps metadata about coding runs. This is useful for
reproducibility: model, codebook, parameters, batch settings, and parent
runs can all be tracked.

``` r
trail <- qlm_trail(coded_leaders_ollama)

trail

# In the current version of quallmer, the export destination is called `file`.
qlm_trail_export(
  trail,
  file = "Labs/Lab_10/leader_credit_blame_coding_trail.json"
)
```

For your own work, save the codebook, model settings, prompts,
validation data, and final coded output. The exact text of the codebook
matters as much as the model name.

``` r
report_checklist <- tibble::tribble(
  ~item, ~why_it_matters,
  "Concept and codebook", "Defines credit claiming and blame attribution",
  "Structured output schema", "Defines allowed labels and scales",
  "Leader corpus and sampling", "Defines the population of speeches being compared",
  "Model/provider/version", "Affects reproducibility and performance",
  "Local model or API model", "Affects reproducibility, privacy, cost, and access",
  "Temperature and other parameters", "Affects stability and variation",
  "Input truncation or chunking rules", "Affects what the model can see",
  "Human validation sample design", "Shows how accuracy was assessed",
  "Validation metrics by label and leader", "Reveals where the model works or fails",
  "Examples of errors", "Reveals conceptual and practical failure modes"
)

report_checklist
```

    # A tibble: 10 × 2
       item                                   why_it_matters                        
       <chr>                                  <chr>                                 
     1 Concept and codebook                   Defines credit claiming and blame att…
     2 Structured output schema               Defines allowed labels and scales     
     3 Leader corpus and sampling             Defines the population of speeches be…
     4 Model/provider/version                 Affects reproducibility and performan…
     5 Local model or API model               Affects reproducibility, privacy, cos…
     6 Temperature and other parameters       Affects stability and variation       
     7 Input truncation or chunking rules     Affects what the model can see        
     8 Human validation sample design         Shows how accuracy was assessed       
     9 Validation metrics by label and leader Reveals where the model works or fails
    10 Examples of errors                     Reveals conceptual and practical fail…

## Extensions

Useful extensions for a final project:

- compare local Ollama coding to an API model;
- test whether results change when leader and party metadata are
  included or excluded;
- compare credit/blame rates between government and opposition leaders;
- code the same speeches with two versions of the codebook;
- compare LLM-coded credit/blame claims with a simpler dictionary
  baseline;
- use human-coded cases to refine codebook instructions before scaling
  up.

## Optional: Using OpenAI Instead Of Ollama

Ollama is useful because it can run locally without an API key. For
larger or more reliable models, you may want to use an API provider
instead. With OpenAI, create an API key in the OpenAI platform dashboard
and store it as an environment variable called `OPENAI_API_KEY`.
OpenAI’s quickstart explains this setup, and the API-key help page
explains where keys are created and why they should not be shared.

``` r
# In your terminal, not inside the Quarto document:
# export OPENAI_API_KEY="your_api_key_here"
```

Once the key is available to R, the `ellmer` syntax is very similar:

``` r
chat <- ellmer::chat_openai(
  model = "gpt-4o-mini",
  params = ellmer::params(temperature = 0)
)

chat$chat(
  "In one sentence, distinguish credit claiming from blame attribution."
)
```

The corresponding `quallmer` call only changes the `model` argument:

``` r
coded_leaders_openai <- qlm_code(
  ollama_demo_texts,
  credit_blame_codebook,
  model = "openai/gpt-4o-mini",
  params = ellmer::params(temperature = 0),
  name = "leader_credit_blame_gpt4o_mini_temperature0_demo"
)

coded_leaders_openai
```

Official OpenAI guidance:
<https://platform.openai.com/docs/quickstart/make-your-first-api-request>
and
<https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key>.

## Practice Exercises

1.  Draft concise coding instructions for a binary task: whether a
    speech contains blame attribution. Store them in an object called
    `binary_blame_instructions`.

``` r
# Your answer here
```

2.  Define a structured schema for the binary blame task with fields
    `blame_attribution`, `confidence`, and `evidence`. Then combine the
    instructions and schema in a `quallmer` codebook called
    `binary_blame_codebook`.

``` r
# Your answer here
```

3.  Show the `qlm_code()` call you would use to code the first five
    leader speeches with this binary blame codebook.

``` r
# Your answer here
```

4.  Create a small mock human-coded data frame with `.id` and
    `blame_attribution` for the first eight speeches.

``` r
# Your answer here
```

5.  Show how you would convert those human codes with `qlm_humancoded()`
    and validate an LLM-coded result with `qlm_validate()`.

``` r
# Your answer here
```

6.  Using `coded_leaders_ollama`, calculate the share of speeches with
    credit claims and blame attributions by `leader`.

``` r
# Your answer here
```

7.  Inspect three speeches where the Ollama output labels the claim type
    as `neither`. What information would you need to decide whether that
    label is correct?

``` r
# Your answer here
```

8.  Reflection: what should be included in a methods section when
    reporting LLM-assisted coding of credit claiming and blame
    attribution with `quallmer` and `ellmer`?

``` r
# Your notes here
```
