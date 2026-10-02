"""The wording of every generated question, and the XBRL concepts behind it.

Positive questions come from ``POSITIVE_CONCEPTS``: twelve us-gaap concepts,
each with the tags companies use for it (tried in order) and four phrasings.
The ``wrong_year`` and ``other_company`` negatives reuse those phrasings with
another year or another company, so each negative differs from a positive in
one thing only.

``SECTOR_CONCEPTS`` feed the ``unreported_concept`` negatives: concepts a bank,
an insurer or an oil company reports and most others never do. Every one of
them is reported by at least one company of the golden set (a test holds this),
so none is an invented tag. ``OFF_DOMAIN`` questions have nothing to do with a
filing; each carries a marker phrase the filing must not contain.

No phrasing uses the relevance model's training template ("Does the passage
answer this question") and none contains a year: the year is always a slot.
"""

from dataclasses import dataclass

FLOW = "flow"
INSTANT = "instant"
USD = "USD"
USD_PER_SHARE = "USD/shares"


@dataclass(frozen=True, slots=True)
class PositiveConcept:
    key: str
    tags: tuple[str, ...]
    unit: str
    kind: str
    phrasings: tuple[str, str, str, str]

    @property
    def per_share(self) -> bool:
        return self.unit == USD_PER_SHARE


@dataclass(frozen=True, slots=True)
class SectorConcept:
    key: str
    tag: str
    label: str
    absent_phrases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OffDomainQuestion:
    key: str
    text: str
    marker: str

    @property
    def names_company(self) -> bool:
        return "{company}" in self.text


POSITIVE_CONCEPTS: tuple[PositiveConcept, ...] = (
    PositiveConcept(
        "revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
        USD,
        FLOW,
        (
            "What was {company}'s total revenue in fiscal {year}?",
            "How much revenue did {company} report for fiscal year {year}?",
            "What were {company}'s total net sales or revenues for fiscal {year}?",
            "Report {company}'s revenue for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "net_income",
        ("NetIncomeLoss",),
        USD,
        FLOW,
        (
            "What was {company}'s net income in fiscal {year}?",
            "How much did {company} earn in net income for fiscal year {year}?",
            "What net income did {company} report for fiscal {year}?",
            "State {company}'s net income for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "operating_income",
        ("OperatingIncomeLoss",),
        USD,
        FLOW,
        (
            "What was {company}'s operating income in fiscal {year}?",
            "How much operating income did {company} report for fiscal year {year}?",
            "What did {company} earn from operations in fiscal {year}?",
            "State {company}'s income from operations for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "research_and_development",
        (
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ),
        USD,
        FLOW,
        (
            "How much did {company} spend on research and development in fiscal {year}?",
            "What was {company}'s R&D expense for fiscal year {year}?",
            "What research and development expense did {company} report for fiscal {year}?",
            "State {company}'s research and development costs for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "selling_general_administrative",
        ("SellingGeneralAndAdministrativeExpense",),
        USD,
        FLOW,
        (
            "What were {company}'s selling, general and administrative expenses in fiscal {year}?",
            "How much SG&A expense did {company} report for fiscal year {year}?",
            "What did {company} spend on selling, general and administrative costs"
            " in fiscal {year}?",
            "State {company}'s SG&A for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "diluted_eps",
        ("EarningsPerShareDiluted",),
        USD_PER_SHARE,
        FLOW,
        (
            "What were {company}'s diluted earnings per share in fiscal {year}?",
            "What diluted EPS did {company} report for fiscal year {year}?",
            "How much did {company} earn per diluted share in fiscal {year}?",
            "State {company}'s diluted net income per share for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "total_assets",
        ("Assets",),
        USD,
        INSTANT,
        (
            "What were {company}'s total assets at the end of fiscal {year}?",
            "How much did {company} report in total assets for fiscal year {year}?",
            "What total assets did {company}'s balance sheet show at fiscal {year} year end?",
            "State {company}'s total assets as of the end of its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "total_liabilities",
        ("Liabilities",),
        USD,
        INSTANT,
        (
            "What were {company}'s total liabilities at the end of fiscal {year}?",
            "How much did {company} report in total liabilities for fiscal year {year}?",
            "What total liabilities did {company}'s balance sheet show at fiscal {year} year end?",
            "State {company}'s total liabilities as of the end of its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "cash",
        ("CashAndCashEquivalentsAtCarryingValue",),
        USD,
        INSTANT,
        (
            "How much cash and cash equivalents did {company} hold at the end of fiscal {year}?",
            "What was {company}'s cash and cash equivalents balance for fiscal year {year}?",
            "What cash and equivalents did {company} report at fiscal {year} year end?",
            "State {company}'s cash and cash equivalents as of the end of its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "long_term_debt",
        ("LongTermDebtNoncurrent", "LongTermDebt"),
        USD,
        INSTANT,
        (
            "How much long-term debt did {company} have at the end of fiscal {year}?",
            "What was {company}'s long-term debt for fiscal year {year}?",
            "What long-term debt did {company} report at fiscal {year} year end?",
            "State {company}'s long-term debt as of the end of its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "capital_expenditures",
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
        USD,
        FLOW,
        (
            "How much did {company} spend on capital expenditures in fiscal {year}?",
            "What were {company}'s payments for property, plant and equipment"
            " in fiscal year {year}?",
            "What capital expenditures did {company} report for fiscal {year}?",
            "State {company}'s purchases of property and equipment for its fiscal {year}.",
        ),
    ),
    PositiveConcept(
        "dividends_paid",
        ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
        USD,
        FLOW,
        (
            "How much did {company} pay in dividends in fiscal {year}?",
            "What were {company}'s dividend payments for fiscal year {year}?",
            "What cash dividends did {company} pay to shareholders in fiscal {year}?",
            "State the dividends {company} paid during its fiscal {year}.",
        ),
    ),
)

SECTOR_CONCEPTS: tuple[SectorConcept, ...] = (
    SectorConcept(
        "net_interest_income",
        "InterestIncomeExpenseNet",
        "net interest income",
        ("net interest income",),
    ),
    SectorConcept(
        "premiums_earned",
        "PremiumsEarnedNet",
        "net premiums earned",
        ("premiums earned", "earned premiums"),
    ),
    SectorConcept(
        "policyholder_benefits",
        "PolicyholderBenefitsAndClaimsIncurredNet",
        "policyholder benefits and claims incurred",
        ("policyholder benefits", "claims incurred"),
    ),
    SectorConcept(
        "loan_loss_provision",
        "ProvisionForLoanLeaseAndOtherLosses",
        "provision for loan losses",
        ("provision for loan losses", "provision for loan and lease losses", "loan losses"),
    ),
    SectorConcept(
        "total_deposits",
        "Deposits",
        "total customer deposits",
        ("total deposits", "customer deposits"),
    ),
    SectorConcept(
        "exploration_expense",
        "ExplorationExpense",
        "exploration expense",
        ("exploration expense", "exploration expenses", "exploration costs"),
    ),
    SectorConcept(
        "noninterest_income",
        "NoninterestIncome",
        "noninterest income",
        ("noninterest income", "non interest income"),
    ),
    SectorConcept(
        "franchise_revenue",
        "FranchiseRevenue",
        "franchise revenue",
        ("franchise revenue", "franchise revenues", "franchised revenues"),
    ),
    SectorConcept(
        "oil_and_gas_revenue",
        "OilAndGasRevenue",
        "oil and gas revenue",
        ("oil and gas revenue", "oil and gas revenues"),
    ),
    SectorConcept(
        "investment_banking_revenue",
        "InvestmentBankingRevenue",
        "investment banking revenue",
        ("investment banking",),
    ),
    SectorConcept(
        "deposit_interest_expense",
        "InterestExpenseDeposits",
        "interest expense on deposits",
        ("interest expense on deposits", "interest on deposits"),
    ),
    SectorConcept(
        "brokerage_commissions",
        "BrokerageCommissionsRevenue",
        "brokerage commissions revenue",
        ("brokerage commissions", "brokerage commission"),
    ),
    SectorConcept(
        "future_policy_benefits",
        "LiabilityForFuturePolicyBenefits",
        "liability for future policy benefits",
        ("future policy benefits",),
    ),
)

SECTOR_PHRASINGS: tuple[str, str, str, str] = (
    "What was {company}'s {label} in fiscal {year}?",
    "How much {label} did {company} report for fiscal year {year}?",
    "According to its annual report, what was {company}'s {label} for fiscal {year}?",
    "State {company}'s {label} for its fiscal {year}.",
)

OFF_DOMAIN: tuple[OffDomainQuestion, ...] = (
    OffDomainQuestion(
        "boiling_point", "What is the boiling point of water in Fahrenheit?", "boiling point"
    ),
    OffDomainQuestion(
        "pride_and_prejudice", "Who wrote the novel Pride and Prejudice?", "pride and prejudice"
    ),
    OffDomainQuestion("jupiter_moons", "How many moons does Jupiter have?", "jupiter"),
    OffDomainQuestion("sourdough", "What is a good recipe for sourdough bread?", "sourdough"),
    OffDomainQuestion(
        "capital_australia", "What is the capital city of Australia?", "capital city of australia"
    ),
    OffDomainQuestion("bicycle_tire", "How do I fix a flat bicycle tire?", "bicycle"),
    OffDomainQuestion("good_morning", "How do you say good morning in Japanese?", "good morning"),
    OffDomainQuestion(
        "common_cold", "What are the usual symptoms of the common cold?", "common cold"
    ),
    OffDomainQuestion(
        "soccer_players", "How many players does a soccer team field at once?", "soccer"
    ),
    OffDomainQuestion("symbol_gold", "What is the chemical symbol for gold?", "chemical symbol"),
    OffDomainQuestion("mona_lisa", "Who painted the Mona Lisa?", "mona lisa"),
    OffDomainQuestion(
        "tallest_mountain", "What is the tallest mountain in Africa?", "tallest mountain"
    ),
    OffDomainQuestion("soft_boiled_egg", "How long should I boil an egg for a soft yolk?", "yolk"),
    OffDomainQuestion("casablanca", "What is the plot of the film Casablanca?", "casablanca"),
    OffDomainQuestion("drop_d", "How do I tune a guitar to drop D?", "guitar"),
    OffDomainQuestion(
        "speed_of_light", "What is the speed of light in a vacuum?", "speed of light"
    ),
    OffDomainQuestion("red_planet", "Which planet is called the red planet?", "red planet"),
    OffDomainQuestion("puppy_sit", "How do I teach a puppy to sit?", "puppy"),
    OffDomainQuestion(
        "photosynthesis", "How does photosynthesis work in plants?", "photosynthesis"
    ),
    OffDomainQuestion("chess_opening", "What is the best chess opening for beginners?", "chess"),
    OffDomainQuestion(
        "favorite_color",
        "What is the favorite color of {company}'s chief executive?",
        "favorite color",
    ),
    OffDomainQuestion("poem", "Write a short poem about {company}.", "poem"),
    OffDomainQuestion(
        "price_next_friday", "What will {company}'s share price be next Friday?", "next friday"
    ),
    OffDomainQuestion(
        "holiday_party", "What is the dress code at {company}'s holiday party?", "holiday party"
    ),
    OffDomainQuestion(
        "ceo_breakfast",
        "What did {company}'s chief executive eat for breakfast today?",
        "breakfast",
    ),
    OffDomainQuestion(
        "cafeteria_lasagna", "Is the lasagna in {company}'s cafeteria any good?", "lasagna"
    ),
    OffDomainQuestion(
        "founder_horoscope", "What is the horoscope sign of {company}'s founder?", "horoscope"
    ),
    OffDomainQuestion(
        "hold_music", "Which hold music plays when you call {company}'s support line?", "hold music"
    ),
    OffDomainQuestion(
        "wifi_password", "What is the Wi-Fi password at {company}'s headquarters?", "wi-fi password"
    ),
    OffDomainQuestion("joke", "Tell me a joke about {company}.", "joke"),
    OffDomainQuestion(
        "office_dog", "What is the name of the office dog at {company}?", "office dog"
    ),
    OffDomainQuestion("ascii_logo", "Draw {company}'s logo in ASCII art.", "ascii"),
    OffDomainQuestion(
        "cat_birthday", "When is the birthday of {company}'s chief executive's cat?", "birthday"
    ),
    OffDomainQuestion(
        "vacation_spot", "Recommend a vacation spot near {company}'s headquarters.", "vacation spot"
    ),
    OffDomainQuestion(
        "daily_walk", "How far does {company}'s chief executive walk each day?", "walk each day"
    ),
    OffDomainQuestion(
        "weekend_movie",
        "Which movie should {company}'s employees watch this weekend?",
        "this weekend",
    ),
    OffDomainQuestion(
        "golf_handicap",
        "What is the golf handicap of {company}'s chief executive?",
        "golf handicap",
    ),
    OffDomainQuestion("haiku", "Compose a haiku about {company}'s products.", "haiku"),
    OffDomainQuestion(
        "pizza_toppings",
        "Which pizza toppings does {company}'s board of directors prefer?",
        "pizza",
    ),
    OffDomainQuestion(
        "ceo_playlist", "What songs are on the playlist of {company}'s chief executive?", "playlist"
    ),
)


def possessive(name: str) -> str:
    """``Apple`` becomes ``Apple's``; ``McDonald's`` stays as it is."""
    return name if name.endswith("'s") else f"{name}'s"


def fill(phrasing: str, *, company: str, year: int | None = None, label: str = "") -> str:
    """A phrasing with its slots filled; ``{company}'s`` takes the right possessive."""
    text = phrasing.replace("{company}'s", possessive(company)).replace("{company}", company)
    text = text.replace("{label}", label)
    return text if year is None else text.replace("{year}", str(year))
