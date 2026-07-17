# QTA Lab 01 Answers: Introduction to R and the House of Commons Corpus


## Learning goals

In this first lab, you will learn how to:

- use Positron to run R code;
- create and inspect basic R objects;
- distinguish numeric, character, logical, and date variables;
- load R packages;
- read in and inspect a small sample of House of Commons speeches;
- treat text as data using string operations;
- make a first plot from political text data.

Throughout the course, we will use a sample of approximately 5,000 House
of Commons speeches from 1945 to 2025. Each row in the dataset is a
speech. The original dataset is much larger (and ranges from 1919-2025)
and was put together by Stefan Couperus and Martijn Schoonvelde, but we
will use a smaller sample for teaching purposes.

## Positron

We will use `R` in Positron. Positron is an integrated development
environment, or IDE, for data science work. It helps you write code, run
code, inspect objects, view plots, and work with files in one place.
Positron also allows you to switch between R and Python, and to use R
Markdown and Quarto documents for reproducible research. If you have
used other IDEs, such as RStudio, you will find Positron familiar, but
otherwise it may take some getting used to (but you will!). Positron is
free to use and available for Windows, Mac, and Linux. You can download
it from [here](https://positron.posit.co/download.html).

For this course, open the main course folder in Positron rather than
opening individual lab folders. This folder is a Quarto project, which
means that code is evaluated relative to the main course folder when the
labs are rendered. Using one project folder also keeps file paths
simple: paths such as `Data/hc_sample_1945_2025.rds` start from
the main course folder.

The interface will usually contain these areas:

- Activity bar, where you can switch between the file explorer, search,
  source control, and extensions.
- Primary side bar, where you can navigate files and folders, search for
  text, and manage source control.
- The Console, where you can run R commands directly.
- The Editor, where you write and save code in files such as `.R` and
  `.qmd` files.
- The Variables or Environment pane, where you can inspect objects you
  have created.
- The Plots, Files, Help, or Viewer panes, where you can inspect output,
  navigate files, and read documentation.

I recommend you take some time to familiarise yourself with Positron.
You can find more information about Positron in the [Positron
documentation](https://positron.posit.co/docs/).

## Running code

Each lab file is a Quarto document, which is a type of R Markdown file.
You can run code in a Quarto document in several ways. You can run the
whole document interactively by clicking the **Run all** button at the
top of the editor. This runs the code chunks in your current R session.
You can also run individual code chunks by clicking the **Run** button
attached to each chunk. To create the final rendered lab document, use
**Render**. You can run one line of code from a `.qmd` file by placing
your cursor on that line and pressing `Cmd + Enter` on Mac or
`Ctrl + Enter` on Windows.

In these lab documents, code appears in grey blocks called code chunks.
Lines that begin with `#` are comments. They are not executed by R, but
they explain what the code is doing.

## R as a calculator

Let’s start very simply: R can be used as a calculator. This is a useful
way to get comfortable with the Console.

``` r
# Addition
2 + 2
```

    [1] 4

``` r
# Exponentiation
3^2
```

    [1] 9

``` r
# Subtraction
4 - 5
```

    [1] -1

``` r
# Roughly how many speeches per year are in our teaching sample?
5000 / 81
```

    [1] 61.7284

R also contains many built-in functions. Functions take inputs, called
arguments, and return outputs. For example, the `mean()` function takes
a numeric vector as input and returns the mean of that vector. The
`median()` function returns the median, and the `sqrt()` function
returns the square root. In the code below, `c()` is a function that
creates a vector from the numbers inside the parentheses.

``` r
mean(c(2, 4))
```

    [1] 3

``` r
median(c(2, 4, 100))
```

    [1] 4

``` r
sqrt(81)
```

    [1] 9

## Objects

We can save results as objects using the assignment operator `<-`.

As you will see later, the House of Commons sample contains a variable
called `terms`, which gives the number of terms in each speech. Let us
make a small vector of speech lengths by hand from five speeches.

``` r
speech_terms <- c(72, 53, 29, 30, 50)
speech_terms
```

    [1] 72 53 29 30 50

Once the object exists, we can use it in later commands.

``` r
mean(speech_terms)
```

    [1] 46.8

``` r
max(speech_terms)
```

    [1] 72

``` r
min(speech_terms)
```

    [1] 29

R objects can contain different types of data. Text is usually stored as
a character vector. Let’s say we have the names of five speakers in the
sample. We can create a character vector using `c()`.

``` r
speakers <- c(
  "Mr. Johnstone",
  "Mr. Woodburn",
  "Mr. Eden",
  "Vice-Admiral Taylor",
  "Sir M. Sueter"
)

speakers
```

    [1] "Mr. Johnstone"       "Mr. Woodburn"        "Mr. Eden"           
    [4] "Vice-Admiral Taylor" "Sir M. Sueter"      

``` r
class(speakers)
```

    [1] "character"

What happens if we try to calculate the mean of a vector of speaker
names?

``` r
# mean(speakers)
```

The line is commented out because it will produce an error. That error
is useful: it tells us that `mean()` expects numeric input, but speaker
names are character data.

## Data types

The most common data types we will use in this course are:

- `numeric`: numbers, such as `72` or `196.6`;
- `integer`: whole numbers, such as a speech number;
- `character`: text, such as a speaker name or speech;
- `logical`: `TRUE` or `FALSE`;
- `Date`: calendar dates.

You can inspect the type of an object using `class()`.

``` r
speech_date <- as.Date("1945-01-16")
mentions_trade <- TRUE

class(speech_terms)
```

    [1] "numeric"

``` r
class(speakers)
```

    [1] "character"

``` r
class(speech_date)
```

    [1] "Date"

``` r
class(mentions_trade)
```

    [1] "logical"

## Packages

Base R contains many functions, but much of the R ecosystem is organised
around packages. Packages are collections of functions, data, and
documentation that have been produced by the R community. You can
install packages from CRAN, the Comprehensive R Archive Network, or from
other sources such as GitHub. Once a package is installed, you can load
it into your R session using `library()`.

Today we will use three packages:

- `dplyr` for working with data frames;
- `stringr` for working with text strings;
- `ggplot2` for plotting.

You only need to install a package once, but you need to load it every
time you start a new R session.

``` r
# install.packages("dplyr")
# install.packages("stringr")
# install.packages("ggplot2")
```

``` r
library(dplyr)
library(stringr)
library(ggplot2)
```

It is good practice to cite packages used in an analysis. You can see
how to cite a package using `citation()`.

``` r
citation("stringr")
```

    To cite package 'stringr' in publications use:

      Wickham H (2023). _stringr: Simple, Consistent Wrappers for Common
      String Operations_. R package version 1.5.1,
      <https://CRAN.R-project.org/package=stringr>.

    A BibTeX entry for LaTeX users is

      @Manual{,
        title = {stringr: Simple, Consistent Wrappers for Common String Operations},
        author = {Hadley Wickham},
        year = {2023},
        note = {R package version 1.5.1},
        url = {https://CRAN.R-project.org/package=stringr},
      }

## Read the House of Commons sample

The full House of Commons dataset is large (6,821,869 speeches). For
teaching purposes, we will use a smaller sample covering 1945 to 2025.

This sample has already been cleaned in two ways. First, very short
speeches and chair contributions have been removed. Second, common party
label variants have been standardised. The original party label is still
available in `party_raw`.

The data file is stored as an `.rds` file, which is a native R file
format. Here we use `readRDS()` to read the file. If our data would have
been stored as a CSV file, we would have used `read.csv()` (base R) or
`read_csv()` (tidyverse) instead.

Because this course folder is a Quarto project, both rendering and
interactive work should use the main course folder as the working
directory. The path below therefore starts from the main course folder.

``` r
data_path <- "Data/hc_sample_1945_2025.rds"

hoc <- readRDS(data_path)
```

Let us inspect the object. The `str()` function is useful when you first
read in a dataset because it shows the variables, their data types, and
a few example values.

``` r
class(hoc)
```

    [1] "data.frame"

``` r
dim(hoc)
```

    [1] 5000   13

``` r
names(hoc)
```

     [1] "date"           "agenda"         "speechnumber"   "speaker"       
     [5] "party"          "party.facts.id" "chair"          "terms"         
     [9] "text"           "speech_url"     "parliament"     "iso3country"   
    [13] "party_raw"     

``` r
str(hoc)
```

    'data.frame':   5000 obs. of  13 variables:
     $ date          : Date, format: "1945-01-16" "1945-01-16" ...
     $ agenda        : chr  "Export Trade" "Fruit and Vegetables (Short Weight)" "BRITISH PRISONERS OF WAR, FAR EAST" "FIGHTING SERVICES (PAY)" ...
     $ speechnumber  : int  1164121 1164131 1166439 1167553 1168346 1171770 1172870 1173225 1173549 1173930 ...
     $ speaker       : chr  "Mr. Johnstone" "Mr. Woodburn" "Mr. Eden" "Vice-Admiral Taylor" ...
     $ party         : chr  "Liberal" "Labour" "Conservative" "Conservative" ...
     $ party.facts.id: num  NA NA NA NA NA NA NA NA NA NA ...
     $ chair         : logi  FALSE FALSE FALSE FALSE FALSE FALSE ...
     $ terms         : num  72 53 29 30 50 489 116 49 31 64 ...
     $ text          : chr  "In addition to discussions with a large number of individual merchants and merchant bankers, many meetings conn"| __truncated__ "asked the President of the Board of Trade whether he is aware of the difficulties encountered by retail fruit a"| __truncated__ "Despite several approaches the Japanese Government have shown themselves completely uninterested in exchanges o"| __truncated__ "I do not think the hon. Member has a very good case. For years past the Socialist Members have consistently vot"| __truncated__ ...
     $ speech_url    : chr  "https://api.parliament.uk/historic-hansard/written_answers/1945/jan/16/export-trade#S5CV0407P0_19450116_CWA_118" "https://api.parliament.uk/historic-hansard/commons/1945/jan/16/fruit-and-vegetables-short-weight#S5CV0407P0_19450116_HOC_181" "https://api.parliament.uk/historic-hansard/written_answers/1945/jan/24/british-prisoners-of-war-far-east#S5CV04"| __truncated__ "https://api.parliament.uk/historic-hansard/commons/1945/jan/30/fighting-services-pay#S5CV0407P0_19450130_HOC_309" ...
     $ parliament    : chr  "UK-HouseOfCommons" "UK-HouseOfCommons" "UK-HouseOfCommons" "UK-HouseOfCommons" ...
     $ iso3country   : chr  "GBR" "GBR" "GBR" "GBR" ...
     $ party_raw     : chr  "Liberal" "Labour" "Conservative" "Conservative" ...
     - attr(*, "sample_seed")= num 20260706
     - attr(*, "sample_frame")= chr "House of Commons speeches, 1945-2025, excluding chair contributions, missing speaker/party/text, and speeches w"| __truncated__
     - attr(*, "source_file")= chr "Data/hc_1919_2025.rds"
     - attr(*, "party_standardisation")= chr "The original party label is stored in party_raw. The party variable harmonises common abbreviations and spelling variants."

The object is a data frame. A data frame is a table where rows are
observations and columns are variables.

The main variables are:

- `date`: date of the speech;
- `agenda`: agenda item or debate heading;
- `speaker`: speaker name;
- `party`: standardised party label;
- `party_raw`: original party label;
- `terms`: number of terms in the speech;
- `text`: speech text;
- `speech_url`: link to the original parliamentary record.

In the next code chunk, we select a few variables and show the first
five rows of the data frame. The “\|\>” operator is called the pipe. It
takes the output of one command and passes it as input to the next
command. So the code below can be read as follows: “Take the `hoc` data
frame, select the `date`, `speaker`, `party`, `agenda`, `terms`, and
`text` variables, and then show the first five rows.”

``` r
hoc |>
  select(date, speaker, party, agenda, terms, text) |>
  head(5)
```

            date             speaker        party
    1 1945-01-16       Mr. Johnstone      Liberal
    2 1945-01-16        Mr. Woodburn       Labour
    3 1945-01-24            Mr. Eden Conservative
    4 1945-01-30 Vice-Admiral Taylor Conservative
    5 1945-01-31       Sir M. Sueter Conservative
                                     agenda terms
    1                          Export Trade    72
    2   Fruit and Vegetables (Short Weight)    53
    3    BRITISH PRISONERS OF WAR, FAR EAST    29
    4               FIGHTING SERVICES (PAY)    30
    5 ROYAL NAVY (GERMAN SUBMARINE WARFARE)    50
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  text
    1 In addition to discussions with a large number of individual merchants and merchant bankers, many meetings connected with our post-war trade have taken place between my Department and Export Groups, a number of which include merchants among their members. The Consultative Committee which advised my Department on its future practice and procedure included a merchant among its members, and I have invited a merchant to join the Overseas Trade Development Council.
    2                                                                                                                                                      asked the President of the Board of Trade whether he is aware of the difficulties encountered by retail fruit and vegetable merchants owing to the absence of any control over the delivery of short weight; and whether he will take steps to ensure the application of weights and measures regulations in this industry.
    3                                                                                                                                                                                                                                                                          Despite several approaches the Japanese Government have shown themselves completely uninterested in exchanges of prisoners of war and have refused even to contemplate an exchange of sick and wounded.
    4                                                                                                                                                                                                                                                                                                       I do not think the hon. Member has a very good case. For years past the Socialist Members have consistently voted against the Estimates, which include the matter of pay——
    5                                                                                                                                                                 asked the First Lord of the Admiralty, in view of the recent pronouncement by General McNaughton, Canadian War Minister, regarding the number of U-boats in the Atlantic, if he can give an assurance that the Allies have sufficient surface vessels and aircraft to deal with this increased submarine menace.

## Inspecting the corpus

We can start with simple descriptive questions. How many speeches are in
the sample? What dates does it cover? Which parties appear most often?

``` r
nrow(hoc)
```

    [1] 5000

``` r
range(hoc$date)
```

    [1] "1945-01-16" "2025-12-16"

``` r
sort(table(hoc$party), decreasing = TRUE)
```


                           Conservative                              Labour 
                                   2412                                2092 
                       Liberal Democrat                 Labour/Co-operative 
                                    167                                  62 
                Scottish National Party                             Liberal 
                                     57                                  56 
                  Ulster Unionist Party           Democratic Unionist Party 
                                     48                                  28 
                       National Liberal                         Independent 
                                     20                                  18 
                            Plaid Cymru  Social Democratic and Labour Party 
                                     12                                   9 
                Social Democratic Party                            National 
                                      5                                   3 
       Green Party of England and Wales Vanguard Unionist Progressive Party 
                                      2                                   2 
     Alliance Party of Northern Ireland    Communist Party of Great Britain 
                                      1                                   1 
               Independent Labour Party                           Reform UK 
                                      1                                   1 
                Republican Labour Party                             Speaker 
                                      1                                   1 
          Ulster Popular Unionist Party 
                                      1 

Because this is a sample, these counts should not be interpreted as the
true number of speeches made by each party in Parliament. They are
useful for learning how to work with text data.

Let us add a year variable and a character-based speech length variable.
The code below uses `mutate()` to create new variables. The `format()`
function extracts the year from the `date` variable, and `nchar()`
counts the number of characters in the speech text.

``` r
hoc <- hoc |>
  mutate(
    year = as.integer(format(date, "%Y")),
    speech_length_chars = nchar(text)
  )

hoc |>
  select(date, year, speaker, party, terms, speech_length_chars) |>
  head(10)
```

             date year             speaker        party terms speech_length_chars
    1  1945-01-16 1945       Mr. Johnstone      Liberal    72                 464
    2  1945-01-16 1945        Mr. Woodburn       Labour    53                 315
    3  1945-01-24 1945            Mr. Eden Conservative    29                 199
    4  1945-01-30 1945 Vice-Admiral Taylor Conservative    30                 170
    5  1945-01-31 1945       Sir M. Sueter Conservative    50                 304
    6  1945-02-14 1945        Mr. Johnston       Labour   489                2835
    7  1945-02-20 1945     Sir J. Anderson     National   116                 727
    8  1945-02-21 1945         Major Mills Conservative    49                 292
    9  1945-02-22 1945           Mr. Viant       Labour    31                 163
    10 1945-02-23 1945      Earl Winterton Conservative    64                 362

## Text as character data

In R, text is usually stored as character data. The `text` column in our
data frame contains the speech itself.

``` r
speech_text <- hoc$text[1:5]

speech_text
```

    [1] "In addition to discussions with a large number of individual merchants and merchant bankers, many meetings connected with our post-war trade have taken place between my Department and Export Groups, a number of which include merchants among their members. The Consultative Committee which advised my Department on its future practice and procedure included a merchant among its members, and I have invited a merchant to join the Overseas Trade Development Council."
    [2] "asked the President of the Board of Trade whether he is aware of the difficulties encountered by retail fruit and vegetable merchants owing to the absence of any control over the delivery of short weight; and whether he will take steps to ensure the application of weights and measures regulations in this industry."                                                                                                                                                     
    [3] "Despite several approaches the Japanese Government have shown themselves completely uninterested in exchanges of prisoners of war and have refused even to contemplate an exchange of sick and wounded."                                                                                                                                                                                                                                                                         
    [4] "I do not think the hon. Member has a very good case. For years past the Socialist Members have consistently voted against the Estimates, which include the matter of pay——"                                                                                                                                                                                                                                                                                                      
    [5] "asked the First Lord of the Admiralty, in view of the recent pronouncement by General McNaughton, Canadian War Minister, regarding the number of U-boats in the Atlantic, if he can give an assurance that the Allies have sufficient surface vessels and aircraft to deal with this increased submarine menace."                                                                                                                                                                

``` r
class(speech_text)
```

    [1] "character"

The `stringr` package gives us useful functions for working with text.

``` r
# Count characters in each speech
str_length(speech_text)
```

    [1] 464 315 199 170 304

``` r
# Convert text to lower case
str_to_lower(speech_text[1])
```

    [1] "in addition to discussions with a large number of individual merchants and merchant bankers, many meetings connected with our post-war trade have taken place between my department and export groups, a number of which include merchants among their members. the consultative committee which advised my department on its future practice and procedure included a merchant among its members, and i have invited a merchant to join the overseas trade development council."

``` r
# Show the first 120 characters of each speech
str_sub(speech_text, start = 1, end = 120)
```

    [1] "In addition to discussions with a large number of individual merchants and merchant bankers, many meetings connected wit"
    [2] "asked the President of the Board of Trade whether he is aware of the difficulties encountered by retail fruit and vegeta"
    [3] "Despite several approaches the Japanese Government have shown themselves completely uninterested in exchanges of prisone"
    [4] "I do not think the hon. Member has a very good case. For years past the Socialist Members have consistently voted agains"
    [5] "asked the First Lord of the Admiralty, in view of the recent pronouncement by General McNaughton, Canadian War Minister,"

We can also detect whether a pattern appears in a speech. This returns a
logical vector: `TRUE` when the pattern is found and `FALSE` when it is
not found.

``` r
str_detect(str_to_lower(speech_text), "trade")
```

    [1]  TRUE  TRUE FALSE FALSE FALSE

``` r
str_detect(str_to_lower(speech_text), "war")
```

    [1]  TRUE  TRUE  TRUE FALSE  TRUE

In political text analysis, this kind of pattern detection is a building
block for more elaborate methods. The difficult part is usually not
counting words. The difficult part is deciding what the words mean in
context.

## A first text-based variable

Suppose we want to know which speeches mention the economy. We can
create a logical variable for this. The code below uses `mutate()` to
create three new variables: `mentions_economy`, `mentions_nhs`, and
`mentions_europe`. Each variable is `TRUE` if the speech contains one of
the relevant keywords, and `FALSE` otherwise. `str_detect()` is used to
detect the keywords, and `str_to_lower()` converts the speech text to
lower case so that the search is case-insensitive. The `\\b` in the
regular expression indicates a word boundary, so that we only match
whole words. We’ll learn more about regular expressions later in the
course so don’t worry if you don’t understand them yet.

``` r
hoc <- hoc |>
  mutate(
    mentions_economy = str_detect(
      str_to_lower(text),
      "\\beconomy\\b|\\beconomic\\b|\\beconomics\\b"
    ),
    mentions_nhs = str_detect(
      str_to_lower(text),
      "\\bnhs\\b|national health service"
    ),
    mentions_europe = str_detect(
      str_to_lower(text),
      "\\beurope\\b|\\beuropean\\b"
    )
  )

hoc |>
  summarise(
    economy_speeches = sum(mentions_economy),
    nhs_speeches = sum(mentions_nhs),
    europe_speeches = sum(mentions_europe)
  )
```

      economy_speeches nhs_speeches europe_speeches
    1              349          107             284

This is a very simple measurement strategy. It will miss some relevant
speeches and include some irrelevant ones. Later in the course, we will
discuss dictionaries, supervised classification, embeddings, and
LLM-assisted annotation as more flexible ways to measure concepts in
text.

## Plotting data

We can use `ggplot2` to plot variables from the data frame. First, let
us plot the most common parties in the sample.

The code below counts the number of speeches by party, filters to
parties with at least 10 speeches, and then creates a horizontal bar
plot. The `reorder()` function is used to order the bars by count.

``` r
party_counts <- hoc |>
  count(party, sort = TRUE) |>
  filter(n >= 10)

party_plot <- ggplot(
  party_counts,
  aes(x = reorder(party, n), y = n)
) +
  geom_col() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Number of speeches",
    title = "Party labels in the House of Commons teaching sample"
  ) +
  theme_minimal()

party_plot
```

![](Lab_Session_QTA_1_Answers_files/figure-commonmark/plot_speeches_by_party-1.png)

We can also inspect how mentions of selected topics vary over time in
the sample. The `group_by()` and `summarise()` functions are used to
calculate the average share of speeches mentioning each topic by year.
We then use `.groups = "drop"` to remove the grouping structure from the
resulting data frame. Finally, we use `ggplot2` to create a line plot
showing the share of speeches mentioning each topic over time.

``` r
topic_by_year <- hoc |>
  group_by(year) |>
  summarise(
    economy = mean(mentions_economy),
    nhs = mean(mentions_nhs),
    europe = mean(mentions_europe),
    .groups = "drop"
  )

topic_plot <- ggplot(topic_by_year, aes(x = year)) +
  geom_line(aes(y = economy, colour = "Economy")) +
  geom_line(aes(y = nhs, colour = "NHS")) +
  geom_line(aes(y = europe, colour = "Europe")) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%")) +
  labs(
    x = NULL,
    y = "Share of sampled speeches",
    colour = NULL,
    title = "Simple keyword mentions over time"
  ) +
  theme_minimal()

topic_plot
```

![](Lab_Session_QTA_1_Answers_files/figure-commonmark/plot_topic_mentions-1.png)

Finally, let us save one of the plots. Because our paths start from the
main course folder, we write the PNG file to the `Lab_sessions/Day_1` folder
explicitly.

``` r
ggsave("Lab_sessions/Day_1/hoc_party_counts.png", plot = party_plot, width = 8, height = 5)
```

## A note on LLMs and text input

Large language models can classify, summarise, and extract information
from text. But they still need carefully designed inputs. Before using
an LLM, we need to decide what information to provide.

Consider one speech from the sample:

``` r
one_speech <- hoc |>
  select(date, speaker, party, agenda, text) |>
  slice(1)

one_speech
```

            date       speaker   party       agenda
    1 1945-01-16 Mr. Johnstone Liberal Export Trade
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  text
    1 In addition to discussions with a large number of individual merchants and merchant bankers, many meetings connected with our post-war trade have taken place between my Department and Export Groups, a number of which include merchants among their members. The Consultative Committee which advised my Department on its future practice and procedure included a merchant among its members, and I have invited a merchant to join the Overseas Trade Development Council.

If we asked an LLM to classify the topic of this speech, should the
prompt include the `speaker`, `party`, `date`, and `agenda`, or only the
speech text? There is no universal answer. Metadata may help the model
interpret the speech, but it may also introduce bias into the
classification.

We will return to these questions later in the course. For now, the
important point is that LLM-based text analysis still depends on the
same research design choices as other QTA methods: unit of analysis,
corpus construction, measurement, validation, and documentation.

## Practice exercises

1.  Create a vector called `parties` that contains the first ten party
    labels in the HoC sample. Inspect its class and length.

``` r
parties <- hoc$party[1:10]

parties
```

     [1] "Liberal"      "Labour"       "Conservative" "Conservative" "Conservative"
     [6] "Labour"       "National"     "Conservative" "Labour"       "Conservative"

``` r
class(parties)
```

    [1] "character"

``` r
length(parties)
```

    [1] 10

2.  Create a new variable called `mentions_immigration` that detects
    speeches containing either `immigration`, `immigrant(s)`, or
    `migrant(s)`.

``` r
hoc <- hoc |>
  mutate(
    mentions_immigration = str_detect(
      str_to_lower(text),
      "\\bimmigration\\b|\\bimmigrant\\b|\\bimmigrants\\b|\\bmigrant\\b|\\bmigrants\\b"
    )
  )

hoc |>
  summarise(immigration_speeches = sum(mentions_immigration))
```

      immigration_speeches
    1                   58

3.  Count how many speeches mention immigration by party. Display only
    parties with at least 10 speeches in the sample.

``` r
hoc |>
  group_by(party) |>
  summarise(
    total_speeches = n(),
    immigration_speeches = sum(mentions_immigration),
    share_immigration = immigration_speeches / total_speeches,
    .groups = "drop"
  ) |>
  filter(total_speeches >= 10) |>
  arrange(desc(immigration_speeches))
```

    # A tibble: 11 × 4
       party                   total_speeches immigration_speeches share_immigration
       <chr>                            <int>                <int>             <dbl>
     1 Conservative                      2412                   30           0.0124 
     2 Labour                            2092                   20           0.00956
     3 Scottish National Party             57                    6           0.105  
     4 National Liberal                    20                    1           0.05   
     5 Ulster Unionist Party               48                    1           0.0208 
     6 Democratic Unionist Pa…             28                    0           0      
     7 Independent                         18                    0           0      
     8 Labour/Co-operative                 62                    0           0      
     9 Liberal                             56                    0           0      
    10 Liberal Democrat                   167                    0           0      
    11 Plaid Cymru                         12                    0           0      

4.  Make a plot showing the average number of terms per speech for the
    six most common parties in the sample.

``` r
top_six_parties <- hoc |>
  count(party, sort = TRUE) |>
  slice_head(n = 6) |>
  pull(party)

average_terms <- hoc |>
  filter(party %in% top_six_parties) |>
  group_by(party) |>
  summarise(
    average_terms = mean(terms),
    .groups = "drop"
  )

ggplot(
  average_terms,
  aes(x = reorder(party, average_terms), y = average_terms)
) +
  geom_col() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Average number of terms",
    title = "Average speech length for the six most common parties"
  ) +
  theme_minimal()
```

![](Lab_Session_QTA_1_Answers_files/figure-commonmark/exercise_average_terms_plot-1.png)

5.  Pick one speech from the dataset. Write down what metadata you would
    include if you wanted to ask an LLM to classify the speech topic,
    and explain why.

``` r
llm_example <- hoc |>
  select(date, speaker, party, agenda, text) |>
  slice(25)

llm_example
```

            date     speaker        party                          agenda
    1 1945-06-07 Mr. Willink Conservative Midwifery Training (Analgesics)
                                                                                                                                                                                                                                                                                                                                   text
    1 Out of 188 local supervising authorities in England and Wales, 101 had made arrangements before 31st December; 1943, for the instruction of their midwives in the administration of analgesics in childbirth. The corresponding figure in the 1938 returns was 30, but there is reason to think that this was an under-statement.

``` r
# For a topic classification task, I would include the speech text and agenda.
# The agenda gives useful debate context, while the text contains the evidence
# that should drive the classification.
#
# I would be cautious about including speaker and party. These metadata may be
# useful for some research questions, but they can also invite the model to infer
# the topic from political identity rather than from the speech content.
#
# I would include the date if the topic scheme depends on historical context.
```
