# GourmetHub Market - Consumer Behavior Analytics

install.packages(c("ggplot2","dplyr","scales","corrplot","gridExtra","caret","forecast","rpart","rpart.plot","randomForest","e1071"))

library(ggplot2)
library(dplyr)
library(scales)
library(corrplot)
library(gridExtra)
library(caret)
library(forecast)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)


# STEP 1: LOAD AND INSPECT

df <- read.csv("Consumer_Data.csv", stringsAsFactors = FALSE)

dim(df)
str(df)
head(df, 5)
colSums(is.na(df))
summary(df)


# STEP 2: DATA QUALITY CHECK
# Checking which columns have missing values
cat("Missing values per column:\n")
print(colSums(is.na(df))[colSums(is.na(df)) > 0])
# Income is the only column with missing values
# Checking how missing Income cases are distributed across Education levels
cat("Education breakdown of missing Income:\n")
table(df$Education[is.na(df$Income)])

# Checking if Z_CostContact and Z_Revenue carry any information
cat("Z_CostContact unique values:", unique(df$Z_CostContact), "\n")
cat("Z_Revenue unique values:", unique(df$Z_Revenue), "\n")

# Checking categorical distributions for invalid or unusual entries
cat("Marital_Status:\n")
table(df$Marital_Status)
cat("Education:\n")
table(df$Education)
cat("Kidhome:\n")
table(df$Kidhome)
cat("Teenhome:\n")
table(df$Teenhome)

# Checking campaign acceptance columns individually
cat("Campaign acceptance:\n")
table(df$AcceptedCmp1)
table(df$AcceptedCmp2)
table(df$AcceptedCmp3)
table(df$AcceptedCmp4)
table(df$AcceptedCmp5)

# Checking Complain before deciding what to do with it
table(df$Complain)
cat("Complain rate:", round(mean(df$Complain) * 100, 1), "% \n")

# Checking zeros in spending columns
cat("Zero counts in spending columns:\n")
sapply(df[, c("MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds")], function(x) sum(x == 0))

# Checking zeros in channel columns
cat("Zero counts in channel columns:\n")
sapply(df[, c("NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases","NumWebVisitsMonth")],function(x) sum(x == 0))

# Checking NumDealsPurchases range
cat("NumDealsPurchases range:\n")
summary(df$NumDealsPurchases)

# STEP 3: FEATURE ENGINEERING

df$Dt_Customer <- as.Date(df$Dt_Customer, format = "%d-%m-%Y")
df$Age <- 2024 - df$Year_Birth
df$Tenure_Days <- as.numeric(as.Date("2014-06-30") - df$Dt_Customer)
df$EnrollYear <- as.integer(format(df$Dt_Customer, "%Y"))
df$EnrollMonth <- format(df$Dt_Customer, "%Y-%m")

# Spending consolidation
# Wines and Meat are the highest-value categories - consolidated into PremiumSpend
# Fruits, Fish, Sweets, Gold are routine categories - consolidated into EverydaySpend
# TotalSpend = all six combined, used as the regression outcome
df$TotalSpend <- df$MntWines + df$MntFruits + df$MntMeatProducts + df$MntFishProducts + df$MntSweetProducts + df$MntGoldProds
df$PremiumSpend <- df$MntWines + df$MntMeatProducts
df$EverydaySpend <- df$MntFruits + df$MntFishProducts + df$MntSweetProducts + df$MntGoldProds

# ChannelCount = number of distinct purchase channels used (0 to 3)
df$ChannelCount <- as.integer(df$NumWebPurchases > 0) + as.integer(df$NumCatalogPurchases > 0) + as.integer(df$NumStorePurchases > 0)

# BasketSize = avg spend per transaction; NA where no purchases recorded
df$BasketSize <- ifelse((df$NumWebPurchases + df$NumCatalogPurchases + df$NumStorePurchases) > 0,
df$TotalSpend / (df$NumWebPurchases + df$NumCatalogPurchases + df$NumStorePurchases), NA)

# LifeStage is for profiling and visualization only
# Kidhome and Teenhome enter models separately - their effects on spending differ
# This will be confirmed in Step 7
df$LifeStage <- case_when( df$Kidhome == 0 & df$Teenhome == 0 ~ "EmptyNester",
df$Kidhome > 0 & df$Teenhome == 0 ~ "YoungFamily",df$Kidhome == 0 & df$Teenhome > 0 ~ "TeenFamily",TRUE ~ "MixedFamily")

# Reviewing extreme deal purchase records before deciding to keep or remove
cat("Records with 10+ deal purchases:\n")
print(df[df$NumDealsPurchases >= 10, c("NumDealsPurchases","Income","TotalSpend","Kidhome","Teenhome")])

# HighValue = 1 if TotalSpend is above the median (upper half of base)
# This creates a balanced binary classification target
# All predictors used later are behavioral and demographic, not derived from TotalSpend
median.spend <- median(df$TotalSpend, na.rm = TRUE)
df$HighValue <- as.integer(df$TotalSpend > median.spend)
cat("HighValue threshold (median TotalSpend): $", median.spend, "\n")
table(df$HighValue)


# STEP 4: DATA CLEANING AND PREPROCESSING

# Printing records with Age above 90 to inspect them
cat("Records with Age above 90:\n")
print(df[df$Age > 90, c("ID","Year_Birth","Age","Income","TotalSpend","Education","Marital_Status")])
# Birth years 1893, 1899, 1900 produce ages above 120
# Maximum verified human lifespan is 122 years - these are data entry errors
df <- df[df$Age <= 90, ]
cat("Records remaining after age removal:", nrow(df), "\n")

# Checking all records with Income above $150,000 before any removal decision
cat("Records with Income above $150,000:\n")
print(df[!is.na(df$Income) & df$Income > 150000, c("ID","Income","TotalSpend","Education","Age","NumCatalogPurchases")])
# Record ID 9432: Income = $666,666 with TotalSpend of only $62
# 28 standard deviations above the mean - no other record is above $162,397
# All other records above $150,000 are consistent and kept
df <- df[is.na(df$Income) | df$Income <= 200000, ]
cat("Records remaining after income outlier removal:", nrow(df), "\n")

# Marital_Status: "Alone", "Absurd", "YOLO" are invalid - recoded to "Single"
# "Together" recoded to "Married" - same household structure and spending context
df$Marital_Status[df$Marital_Status %in% c("Alone","Absurd","YOLO")] <- "Single"
df$Marital_Status[df$Marital_Status == "Together"] <- "Married"
cat("Marital_Status after recoding:\n")
table(df$Marital_Status)

# Imputing missing Income using median within each Education group
# I used group median because it is better than global median since income differs by education level
df <- df %>%
  group_by(Education) %>%
  mutate(Income = ifelse(is.na(Income), median(Income, na.rm = TRUE), Income)) %>%
  ungroup() %>%
  as.data.frame()
cat("Missing Income after imputation:", sum(is.na(df$Income)), "\n")

# Education as ordered factor - five levels representing a genuine hierarchy
# Excluded from k-means because Euclidean distance assumes equal spacing between levels
df$Education <- factor(df$Education, levels = c("Basic","2n Cycle","Graduation","Master","PhD"), ordered = TRUE)

# Marital_Status as unordered nominal factor
df$Marital_Status <- factor(df$Marital_Status)

# Dropping columns that have no analytical value after feature engineering
# Mnt columns: replaced by TotalSpend, PremiumSpend, EverydaySpend
# Year_Birth, Dt_Customer: replaced by Age, Tenure_Days, EnrollYear, EnrollMonth
drop.cols <- c("Z_CostContact","Z_Revenue","ID","Complain", "MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds",
 "Year_Birth","Dt_Customer","Response")
df <- df[, !names(df) %in% drop.cols]
cat("Final dataset:", nrow(df), "rows,", ncol(df), "columns\n")
names(df)

# STEP 5: SUMMARY STATISTICS

summary(df[, c("Age","Income","Recency","Tenure_Days","TotalSpend","PremiumSpend","EverydaySpend","BasketSize",
  "NumCatalogPurchases","NumWebVisitsMonth","NumStorePurchases","NumWebPurchases","NumDealsPurchases","ChannelCount")])

# Checking skewness to decide whether a transformation is needed
cat("TotalSpend skewness:", round((mean(df$TotalSpend) - median(df$TotalSpend)) / sd(df$TotalSpend), 3), "\n")
cat("Income skewness:", round((mean(df$Income) - median(df$Income)) / sd(df$Income), 3), "\n")
# TotalSpend is moderately right-skewed - residual plots will be checked after regression

table(df$Education)
table(df$Marital_Status)
table(df$LifeStage)
table(df$HighValue)

# Lifecycle stage profile (BP1)
ls.summary <- df %>%
  group_by(LifeStage) %>%
  summarise( n = n(), PctOfBase = round(n() / nrow(df) * 100, 1),
  AvgSpend = round(mean(TotalSpend), 0), SD_Spend = round(sd(TotalSpend), 0),
  PctOfRevenue = round(sum(TotalSpend) / sum(df$TotalSpend) * 100, 1), AvgIncome = round(mean(Income), 0),
  AvgCatalog = round(mean(NumCatalogPurchases), 2), AvgWebVisits = round(mean(NumWebVisitsMonth), 2),
  PctHighValue = round(mean(HighValue) * 100, 1), .groups = "drop")
print(ls.summary)

cat("\nHighValue rate by LifeStage:\n")
print(round(prop.table(table(df$LifeStage, df$HighValue), margin = 1) * 100, 1))

# Cohort quality comparison (BP2)
cohort.summary <- df %>%
group_by(EnrollYear) %>%
summarise(n = n(), AvgSpend = round(mean(TotalSpend), 0), SD_Spend = round(sd(TotalSpend), 0),
AvgCatalog = round(mean(NumCatalogPurchases), 2),PctHighValue = round(mean(HighValue) * 100, 1),.groups = "drop")
print(cohort.summary)

# HighValue vs Standard consumer profile (BP2)
cat("\nHighValue vs Standard consumer profile:\n")
hv.profile <- df %>%
  group_by(HighValue) %>%
  summarise(n = n(),AvgSpend = round(mean(TotalSpend), 0), AvgIncome = round(mean(Income), 0),
  AvgRecency = round(mean(Recency), 0),AvgCatalog = round(mean(NumCatalogPurchases), 2),
  AvgBasketSize = round(mean(BasketSize, na.rm = TRUE), 0),AvgTenure = round(mean(Tenure_Days), 0),
  PctCmp1 = round(mean(AcceptedCmp1) * 100, 1),PctCmp2 = round(mean(AcceptedCmp2) * 100, 1),
  PctCmp3 = round(mean(AcceptedCmp3) * 100, 1),PctCmp4 = round(mean(AcceptedCmp4) * 100, 1),
  PctCmp5 = round(mean(AcceptedCmp5) * 100, 1), .groups = "drop")
print(hv.profile)

cat("\nCatalog users (n =", sum(df$NumCatalogPurchases > 0), "):",
    "mean $", round(mean(df$TotalSpend[df$NumCatalogPurchases > 0]), 0), "\n")
cat("No catalog (n =", sum(df$NumCatalogPurchases == 0), "):",
    "mean $", round(mean(df$TotalSpend[df$NumCatalogPurchases == 0]), 0), "\n")


# STEP 6: EDA VISUALIZATIONS

# Univariate boxplots for key continuous variables
par(mfrow = c(2, 4))
boxplot(df$Age, main = "Age", ylab = "Years")
boxplot(df$Income, main = "Income", ylab = "$")
boxplot(df$TotalSpend, main = "Total Spend", ylab = "$")
boxplot(df$Recency, main = "Recency", ylab = "Days")
boxplot(df$NumCatalogPurchases, main = "Catalog Purchases", ylab = "Count")
boxplot(df$NumWebVisitsMonth, main = "Web Visits/Month", ylab = "Count")
boxplot(df$NumStorePurchases, main = "Store Purchases", ylab = "Count")
boxplot(df$BasketSize, main = "Basket Size", ylab = "$/transaction")
par(mfrow = c(1, 1))

# TotalSpend distribution with HighValue threshold
ggplot(df, aes(x = TotalSpend)) +
  geom_histogram(bins = 40, fill = "#7F77DD", color = "white") +
  geom_vline(xintercept = median.spend, color = "#D85A30", linewidth = 1, linetype = "dashed") +
  annotate("text", x = median.spend + 60, y = 180, hjust = 0, size = 3.5,
  label = paste0("Median = $", round(median.spend))) + scale_x_continuous(labels = dollar_format()) +
  labs(title = "Total Spend Distribution with HighValue threshold", x = "Total Spend ($)", y = "Count")

# HighValue class balance
ggplot(df, aes(x = factor(HighValue, labels = c("Standard","HighValue")))) +
geom_bar(fill = c("#D3D1C7","#0F6E56"), width = 0.5) +
geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4, size = 4) +
labs(title = "Classification Target: balanced 50/50 split", x = "", y = "Count")

# BP1: Average spend by lifecycle stage
ls.plot <- ls.summary %>%
  mutate(LifeStage = factor(LifeStage, levels = c("YoungFamily","MixedFamily","TeenFamily","EmptyNester")))

ggplot(ls.plot, aes(x = LifeStage, y = AvgSpend, fill = LifeStage)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(AvgSpend)), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("#D3D1C7","#C9B8E8","#9FE1CB","#0F6E56")) +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 1300)) +
  labs(title = "BP1: Average spend rises as children leave home", x = "Lifecycle Stage", y = "Avg Total Spend ($)")

ggplot(ls.plot, aes(x = LifeStage, y = PctHighValue, fill = LifeStage)) +
geom_col(width = 0.6, show.legend = FALSE) + geom_text(aes(label = paste0(PctHighValue, "%")), vjust = -0.4, size = 4) +
scale_fill_manual(values = c("#D3D1C7","#C9B8E8","#9FE1CB","#0F6E56")) + scale_y_continuous(limits = c(0, 80)) +
labs(title = "High-value consumer rate by lifecycle stage", x = "Lifecycle Stage", y = "% HighValue")

par(mfrow = c(1, 2))
boxplot(TotalSpend ~ Kidhome, data = df, main = "Spend by Number of Young Children",
  xlab = "Young Children at Home", ylab = "Total Spend ($)", col = c("#9FE1CB","#F5C4B3","#D85A30"))
boxplot(TotalSpend ~ Teenhome, data = df, main = "Spend by Number of Teenagers",
  xlab = "Teenagers at Home", ylab = "Total Spend ($)", col = c("#9FE1CB","#F5C4B3","#D85A30"))
par(mfrow = c(1, 1))
# Young children suppression looks stronger than teenagers

# Average spend by Education and Marital Status
edu.plot <- df %>% group_by(Education) %>% summarise(AvgSpend = mean(TotalSpend), .groups = "drop")
mar.plot <- df %>% group_by(Marital_Status) %>% summarise(AvgSpend = mean(TotalSpend), .groups = "drop")

p.edu <- ggplot(edu.plot, aes(x = Education, y = AvgSpend, fill = Education)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(round(AvgSpend))), vjust = -0.4, size = 3.5) +
  scale_fill_brewer(palette = "Purples") +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 800)) +
  labs(title = "Avg Spend by Education", x = "", y = "Avg Spend ($)")

p.mar <- ggplot(mar.plot, aes(x = Marital_Status, y = AvgSpend, fill = Marital_Status)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(round(AvgSpend))), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = c("#0F6E56","#7F77DD","#D85A30","#9FE1CB")) +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 800)) +
  labs(title = "Avg Spend by Marital Status", x = "", y = "Avg Spend ($)")

grid.arrange(p.edu, p.mar, ncol = 2)

# BP2: Cohort spend and catalog decline
p.coh1 <- ggplot(cohort.summary, aes(x = factor(EnrollYear), y = AvgSpend, fill = factor(EnrollYear))) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(AvgSpend)), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("2012"="#0F6E56","2013"="#7F77DD","2014"="#D85A30")) +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 900)) +
  labs(title = "Avg Spend by Enrollment Cohort", x = "Enrollment Year", y = "Avg Spend ($)")

p.coh2 <- ggplot(cohort.summary, aes(x = factor(EnrollYear), y = AvgCatalog, fill = factor(EnrollYear))) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(AvgCatalog, 2)), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("2012"="#0F6E56","2013"="#7F77DD","2014"="#D85A30")) +
  scale_y_continuous(limits = c(0, 4)) +
  labs(title = paste0("Avg Catalog Purchases by Cohort (",round((1 - cohort.summary$AvgCatalog[cohort.summary$EnrollYear == 2014] /
  cohort.summary$AvgCatalog[cohort.summary$EnrollYear == 2012]) * 100, 0),"% decline 2012 to 2014)"),
  x = "Enrollment Year", y = "Avg Catalog Purchases")

grid.arrange(p.coh1, p.coh2, ncol = 2)

# Predictors split by HighValue
par(mfrow = c(2, 3))
boxplot(Income ~ HighValue, data = df, xlab = "HighValue", ylab = "$", main = "Income", col = c("#D3D1C7","#0F6E56"))
boxplot(NumCatalogPurchases ~ HighValue, data = df, xlab = "HighValue", ylab = "Count", main = "Catalog Purchases", col = c("#D3D1C7","#0F6E56"))
boxplot(NumWebVisitsMonth ~ HighValue, data = df, xlab = "HighValue", ylab = "Count", main = "Web Visits", col = c("#D3D1C7","#0F6E56"))
boxplot(NumStorePurchases ~ HighValue, data = df, xlab = "HighValue", ylab = "Count", main = "Store Purchases", col = c("#D3D1C7","#0F6E56"))
boxplot(Recency ~ HighValue, data = df, xlab = "HighValue", ylab = "Days", main = "Recency", col = c("#D3D1C7","#0F6E56"))
boxplot(BasketSize ~ HighValue, data = df, xlab = "HighValue", ylab = "$/tr", main = "Basket Size", col = c("#D3D1C7","#0F6E56"))
par(mfrow = c(1, 1))

# Fig 4: Monthly enrollment trend
enrollment.monthly <- df %>%
  group_by(EnrollMonth) %>%
  summarise(NewConsumers = n(), .groups = "drop") %>%
  filter(EnrollMonth >= "2012-09") %>%
  mutate(Date = as.Date(paste0(EnrollMonth, "-01")))

ggplot(enrollment.monthly, aes(x = Date, y = NewConsumers)) +
  geom_line(color = "#7F77DD", linewidth = 1) + geom_point(color = "#7F77DD", size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "#D85A30", fill = "#F5C4B3", alpha = 0.25, linewidth = 0.9) +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months") +
  labs(title = "Monthly Consumer Enrollment Sep 2012 to Jun 2014",
  x = "Enrollment Month", y = "New Consumers") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Fig 5: Income vs TotalSpend colored by HighValue
ggplot(df, aes(x = Income, y = TotalSpend, color = factor(HighValue, labels = c("Standard","HighValue")))) +
  geom_point(alpha = 0.4, size = 0.9) +
  scale_color_manual(values = c("Standard"="#D3D1C7","HighValue"="#0F6E56")) +
  scale_x_continuous(labels = dollar_format()) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "Income vs Total Spend colored by HighValue",
  subtitle = "HighValue consumers concentrate in the high-income, high-spend region",
  x = "Income ($)", y = "Total Spend ($)", color = "")

# BP3: Scatter plots vs TotalSpend before formal correlation analysis 
par(mfrow = c(2, 2))
plot(df$Income, df$TotalSpend, pch = 19, cex = 0.4, col = "#7F77DD55",
 xlab = "Income ($)", ylab = "Total Spend ($)", main = "TotalSpend vs Income")
abline(lm(TotalSpend ~ Income, data = df), col = "#D85A30", lwd = 2)

plot(df$NumCatalogPurchases, df$TotalSpend, pch = 19, cex = 0.4, col = "#0F6E5655",
 xlab = "Catalog Purchases", ylab = "Total Spend ($)", main = "TotalSpend vs Catalog Purchases")
abline(lm(TotalSpend ~ NumCatalogPurchases, data = df), col = "#D85A30", lwd = 2)

plot(df$NumWebVisitsMonth, df$TotalSpend, pch = 19, cex = 0.4, col = "#D85A3055",
 xlab = "Web Visits/Month", ylab = "Total Spend ($)", main = "TotalSpend vs Web Visits")
abline(lm(TotalSpend ~ NumWebVisitsMonth, data = df), col = "#D85A30", lwd = 2)

plot(df$NumStorePurchases, df$TotalSpend, pch = 19, cex = 0.4, col = "#9F77DD55",
 xlab = "Store Purchases", ylab = "Total Spend ($)", main = "TotalSpend vs Store Purchases")
abline(lm(TotalSpend ~ NumStorePurchases, data = df), col = "#D85A30", lwd = 2)
par(mfrow = c(1, 1))
# Income and catalog show strong positive slopes; web visits shows a negative slope

# Fig 6: Channel breadth and web quintile spend (both address BP3)
channel.plot <- df %>%
  group_by(ChannelCount) %>%
  summarise(AvgSpend = mean(TotalSpend), .groups = "drop")

p.channel <- ggplot(channel.plot, aes(x = factor(ChannelCount), y = AvgSpend, fill = factor(ChannelCount))) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(round(AvgSpend))), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("0"="#D3D1C7","1"="#C9B8E8","2"="#9FE1CB","3"="#0F6E56")) +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 900)) +
  labs(title = "BP3: Avg Spend by Number of Channels Used",
  x = "Number of Channels (ChannelCount)", y = "Avg Total Spend ($)")

web.quintile <- cut(df$NumWebVisitsMonth, breaks = quantile(df$NumWebVisitsMonth, probs = seq(0, 1, 0.2), na.rm = TRUE),
 labels = c("Lowest","Q2","Q3","Q4","Highest"), include.lowest = TRUE)

web.plot <- df %>%
  mutate(WebQ = web.quintile) %>%
  filter(!is.na(WebQ)) %>%
  group_by(WebQ) %>%
  summarise(AvgSpend = mean(TotalSpend), .groups = "drop")

p.web <- ggplot(web.plot, aes(x = WebQ, y = AvgSpend, fill = WebQ)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = dollar(round(AvgSpend))), vjust = -0.4, size = 3.8) +
  scale_fill_manual(values = c("#0F6E56","#5DCAA5","#9FE1CB","#F5C4B3","#D85A30")) +
  scale_y_continuous(labels = dollar_format(), limits = c(0, 1300)) +
  labs(title = "BP3: Average spend decreases as web visit frequency increases",
  x = "Web Visit Quintile", y = "Avg Total Spend ($)")

grid.arrange(p.channel, p.web, ncol = 2)

# Catalog user scatter
p.cat <- ggplot(df, aes(x = Income, y = TotalSpend,
  color = ifelse(NumCatalogPurchases > 0, "Catalog User","No Catalog"))) +
  geom_point(alpha = 0.3, size = 0.8) +
  scale_color_manual(values = c("Catalog User"="#0F6E56","No Catalog"="#D3D1C7")) +
  scale_x_continuous(labels = dollar_format()) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "BP3: Catalog users dominate high-spend region at every income level",
  subtitle = paste0("Avg spend: $", round(mean(df$TotalSpend[df$NumCatalogPurchases > 0]), 0),
  " (catalog) vs $", round(mean(df$TotalSpend[df$NumCatalogPurchases == 0]), 0)," (no catalog)"),
  x = "Income ($)", y = "Total Spend ($)", color = "")
p.cat

# Revenue concentration
df.sorted <- df[order(-df$TotalSpend), ]
df.sorted$cum.cust <- seq_len(nrow(df.sorted)) / nrow(df.sorted) * 100
df.sorted$cum.rev <- cumsum(df.sorted$TotalSpend) / sum(df.sorted$TotalSpend) * 100
top20.rev <- df.sorted$cum.rev[which.min(abs(df.sorted$cum.cust - 20))]

ggplot(df.sorted, aes(x = cum.cust, y = cum.rev)) +
  geom_line(color = "#D85A30", linewidth = 1.1) +
  geom_hline(yintercept = top20.rev, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 20, linetype = "dashed", color = "gray50") +
  annotate("text", x = 22, y = 20, hjust = 0, size = 3.2, color = "gray30",
  label = paste0("Top 20% of consumers = ", round(top20.rev, 0), "% of revenue")) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "Revenue concentration", x = "Cumulative % of Consumers", y = "Cumulative % of Revenue")

# Correlation heatmap with numeric coefficients in each cell
cor.vars <- df[, c("TotalSpend","PremiumSpend","EverydaySpend","Income","Age","Recency",
  "NumCatalogPurchases","NumWebVisitsMonth","NumStorePurchases","NumWebPurchases",
  "NumDealsPurchases","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4",
  "AcceptedCmp5","Kidhome","Teenhome","ChannelCount","Tenure_Days")]
corrplot(cor(cor.vars, use = "complete.obs"),method = "color", type = "upper", order = "hclust",
  addCoef.col = "black", number.cex = 0.42, tl.col = "black", tl.srt = 45, tl.cex = 0.62,
  col = colorRampPalette(c("#D85A30","white","#7F77DD"))(200),title = "Correlation Matrix", mar = c(0, 0, 2, 0))


# STEP 7: PREDICTOR RELEVANCE ANALYSIS

all.class.candidates <- c("Income","NumCatalogPurchases","NumWebVisitsMonth","NumStorePurchases","NumWebPurchases","Kidhome",
"Teenhome","Recency","Age","ChannelCount","Tenure_Days","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5")

all.reg.candidates <- c("Income","NumCatalogPurchases","NumWebVisitsMonth","NumStorePurchases","NumWebPurchases","Kidhome","Teenhome",
"Recency","Age","ChannelCount","EnrollYear","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5")

all.clust.candidates <- c("Income","PremiumSpend","EverydaySpend","Recency","NumCatalogPurchases",
 "NumWebVisitsMonth","NumDealsPurchases","Kidhome","Teenhome","Age","ChannelCount")

cat("Correlations with HighValue (ranked):\n")
hv.corr <- cor(df[, c(all.class.candidates,"HighValue")], use = "complete.obs")[,"HighValue"]
print(round(sort(hv.corr[names(hv.corr) != "HighValue"], decreasing = TRUE), 3))

cat("\nCorrelations with TotalSpend (ranked):\n")
sp.corr <- cor(df[, c(all.reg.candidates,"TotalSpend")], use = "complete.obs")[,"TotalSpend"]
print(round(sort(sp.corr[names(sp.corr) != "TotalSpend"], decreasing = TRUE), 3))

cat("\nClustering variable pairwise correlations:\n")
print(round(cor(df[, all.clust.candidates], use = "complete.obs"), 2))


# STEP 8: DIMENSION REDUCTION 
# Based on correlation rankings above, final predictor sets are selected
# Variables with near-zero correlation with the outcome are dropped

# Classification: AcceptedCmp3 shows near-zero correlation with HighValue - dropped
# Education and Marital_Status added - models handle factors natively
classification.predictors <- c("Income","Education","Marital_Status","NumCatalogPurchases","NumWebVisitsMonth",
  "NumStorePurchases","NumWebPurchases","Kidhome","Teenhome","Recency","Age","ChannelCount","Tenure_Days",
  "AcceptedCmp1","AcceptedCmp2","AcceptedCmp4","AcceptedCmp5")

# Regression: AcceptedCmp3 also shows near-zero correlation with TotalSpend - dropped
# EnrollYear retained to test cohort quality decline seen in EDA
regression.predictors <- c("Income","Education","Marital_Status","NumCatalogPurchases","NumWebVisitsMonth",
  "NumStorePurchases","NumWebPurchases","Kidhome","Teenhome","Recency","Age","ChannelCount","EnrollYear",
  "AcceptedCmp1","AcceptedCmp2","AcceptedCmp4","AcceptedCmp5")

# Clustering: AcceptedCmp columns excluded (binary 0/1 with very low rates)
# Education and Marital_Status excluded (cannot use in k-means without equal-spacing assumption)
# TotalSpend excluded because it correlates with PremiumSpend at r = 0.982
clustering.vars <- c("Income","PremiumSpend","EverydaySpend","Recency","NumCatalogPurchases", 
  "NumWebVisitsMonth","NumDealsPurchases","Kidhome","Teenhome","Age","ChannelCount")

cat("Classification predictors (", length(classification.predictors), "):\n")
print(classification.predictors)
cat("\nRegression predictors (", length(regression.predictors), "):\n")
print(regression.predictors)
cat("\nClustering variables (", length(clustering.vars), "):\n")
print(clustering.vars)


# STEP 9: DATA ENGINEERING AND TRANSFORMATION

# Z-score standardization for k-means only
# Supervised models use the original unstandardized scale
df.cluster <- df[, clustering.vars]
df.cluster.scaled <- as.data.frame(scale(df.cluster))

cat("Column means after scaling (should be near 0):\n")
print(round(colMeans(df.cluster.scaled), 3))
cat("Column SDs after scaling (should be near 1):\n")
print(round(apply(df.cluster.scaled, 2, sd), 3))


# STEP 10: DATA PARTITIONING - 60/20/20 STRATIFIED SPLIT

# Stratified on HighValue to preserve 50/50 class balance in all three sets
set.seed(2026)
train.idx <- createDataPartition(df$HighValue, p = 0.60, list = FALSE)
df.train <- df[train.idx, ]
df.temp <- df[-train.idx, ]

val.idx <- createDataPartition(df.temp$HighValue, p = 0.50, list = FALSE)
df.val <- df.temp[val.idx, ]
df.test <- df.temp[-val.idx, ]

cat("Train:", nrow(df.train), "| Validation:", nrow(df.val), "| Test:", nrow(df.test), "\n")
cat("HighValue % - Train:", round(mean(df.train$HighValue) * 100, 1),
    "| Val:", round(mean(df.val$HighValue) * 100, 1),
    "| Test:", round(mean(df.test$HighValue) * 100, 1), "\n")


# STEP 11: MODEL FITTING

# CLUSTERING - K-MEANS (BP1)

# Elbow method to select k
set.seed(2026)
wss <- sapply(1:8, function(k) {kmeans(df.cluster.scaled, centers = k, nstart = 25)$tot.withinss})
plot(1:8, wss, type = "b", pch = 19, col = "#0F6E56",
     xlab = "Number of Clusters (k)", ylab = "Total Within-Cluster SS",
     main = "Elbow Method - Select Optimal k")

# k = 4 selected: consistent with elbow plot and BP1 lifecycle theory
set.seed(2026)
km.fit <- kmeans(df.cluster.scaled, centers = 4, nstart = 25)
df$Cluster <- factor(km.fit$cluster)
table(df$Cluster)

# Cluster profiles on original scale
cluster.profile <- df %>%
  group_by(Cluster) %>%
  summarise(
    n = n(),
    AvgSpend = round(mean(TotalSpend), 0),
    AvgIncome = round(mean(Income), 0),
    AvgCatalog = round(mean(NumCatalogPurchases), 2),
    AvgWebVisits = round(mean(NumWebVisitsMonth), 2),
    AvgKidhome = round(mean(Kidhome), 2),
    AvgTeenhome = round(mean(Teenhome), 2),
    PctHighValue = round(mean(HighValue) * 100, 1),
    .groups = "drop")
print(cluster.profile)

ggplot(df, aes(x = Cluster, y = TotalSpend, fill = Cluster)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = c("#0F6E56","#7F77DD","#D85A30","#9FE1CB")) +
  scale_y_continuous(labels = dollar_format()) +
  labs(title = "BP1: Total Spend by Cluster", x = "Cluster", y = "Total Spend ($)")

cat("Cluster vs LifeStage:\n")
print(table(df$Cluster, df$LifeStage))


# CLASSIFICATION - HIGHVALUE PREDICTION (BP2)

X.train <- df.train[, classification.predictors]
y.train <- factor(df.train$HighValue, levels = c(0, 1), labels = c("Standard","HighValue"))
X.val <- df.val[, classification.predictors]
y.val <- factor(df.val$HighValue, levels = c(0, 1), labels = c("Standard","HighValue"))
X.test <- df.test[, classification.predictors]
y.test <- factor(df.test$HighValue, levels = c(0, 1), labels = c("Standard","HighValue"))

# Model 1: Logistic Regression
# Outcome renamed to avoid column name collision with predictors
logit.data <- cbind(X.train, Outcome = df.train$HighValue)
logit.fit <- glm(Outcome ~ ., data = logit.data, family = binomial)
summary(logit.fit)

logit.val.prob <- predict(logit.fit, X.val, type = "response")
logit.test.prob <- predict(logit.fit, X.test, type = "response")
logit.val.pred <- factor(ifelse(logit.val.prob > 0.5, "HighValue","Standard"), levels = c("Standard","HighValue"))
logit.test.pred <- factor(ifelse(logit.test.prob > 0.5, "HighValue","Standard"), levels = c("Standard","HighValue"))

cat("\nLogistic Regression - Validation:\n")
cm.logit.val <- confusionMatrix(logit.val.pred, y.val, positive = "HighValue")
print(cm.logit.val)

# Model 2: Classification Tree
tree.fit <- rpart(y.train ~ ., data = X.train, method = "class", control = rpart.control(cp = 0.01, minsplit = 20))
rpart.plot(tree.fit, type = 4, extra = 104, main = "BP2: Classification Tree - HighValue Consumer")

tree.val.pred <- predict(tree.fit, X.val, type = "class")

cat("\nClassification Tree - Validation:\n")
cm.tree.val <- confusionMatrix(tree.val.pred, y.val, positive = "HighValue")
print(cm.tree.val)

# Model 3: Naive Bayes
nb.fit <- naiveBayes(x = X.train, y = y.train)
nb.val.pred <- predict(nb.fit, X.val)

cat("\nNaive Bayes - Validation:\n")
cm.nb.val <- confusionMatrix(nb.val.pred, y.val, positive = "HighValue")
print(cm.nb.val)

# Validation comparison - Sensitivity is the priority metric
# Missing a HighValue consumer (false negative) is more costly than a false alarm
class.results <- data.frame(Model = c("Logistic Regression","Classification Tree","Naive Bayes"),
Accuracy = round(c(cm.logit.val$overall["Accuracy"],cm.tree.val$overall["Accuracy"],
   cm.nb.val$overall["Accuracy"]), 3),
Sensitivity = round(c(cm.logit.val$byClass["Sensitivity"],cm.tree.val$byClass["Sensitivity"],
   cm.nb.val$byClass["Sensitivity"]), 3),
Specificity = round(c(cm.logit.val$byClass["Specificity"],cm.tree.val$byClass["Specificity"],
   cm.nb.val$byClass["Specificity"]), 3))
cat("\nClassification - Validation Comparison:\n")
print(class.results)

# Logistic Regression leads on both Accuracy and Sensitivity
# It also shows which predictors drive classification and by how much - useful for BG2
cat("\nLogistic Regression - Test Set:\n")
cm.logit.test <- confusionMatrix(logit.test.pred, y.test, positive = "HighValue")
print(cm.logit.test)


# REGRESSION - TOTAL SPEND PREDICTION (BP1 and BP3)

X.train.reg <- df.train[, regression.predictors]
y.train.reg <- df.train$TotalSpend
X.val.reg <- df.val[, regression.predictors]
y.val.reg <- df.val$TotalSpend
X.test.reg <- df.test[, regression.predictors]
y.test.reg <- df.test$TotalSpend

# Model 1: Multiple Linear Regression
# Coefficients test whether catalog and web independently predict spending (BP3)
lm.fit <- lm(TotalSpend ~ ., data = cbind(X.train.reg, TotalSpend = y.train.reg))
summary(lm.fit)

par(mfrow = c(2, 2))
plot(lm.fit)
par(mfrow = c(1, 1))

lm.val.pred <- predict(lm.fit, X.val.reg)
lm.val.rmse <- sqrt(mean((y.val.reg - lm.val.pred)^2))
lm.val.r2 <- cor(y.val.reg, lm.val.pred)^2
cat("\nLinear Regression - Validation RMSE:", round(lm.val.rmse, 2), "| R2:", round(lm.val.r2, 3), "\n")

# Model 2: Regression Tree
reg.tree.fit <- rpart(TotalSpend ~ ., data = cbind(X.train.reg, TotalSpend = y.train.reg),
                      method = "anova", control = rpart.control(cp = 0.01, minsplit = 20))
rpart.plot(reg.tree.fit, type = 4, extra = 101, main = "BP3: Regression Tree - Total Spend Segments")

reg.tree.val.pred <- predict(reg.tree.fit, X.val.reg)
reg.tree.val.rmse <- sqrt(mean((y.val.reg - reg.tree.val.pred)^2))
reg.tree.val.r2 <- cor(y.val.reg, reg.tree.val.pred)^2
cat("Regression Tree - Validation RMSE:", round(reg.tree.val.rmse, 2), "| R2:", round(reg.tree.val.r2, 3), "\n")

# Model 3: Random Forest
# Variable importance shows which channel predictor matters most (BP3)
set.seed(2026)
rf.fit <- randomForest(TotalSpend ~ ., data = cbind(X.train.reg, TotalSpend = y.train.reg),ntree = 500, importance = TRUE)
varImpPlot(rf.fit, main = "BP3: Variable Importance - Predictors of TotalSpend")

rf.val.pred <- predict(rf.fit, X.val.reg)
rf.val.rmse <- sqrt(mean((y.val.reg - rf.val.pred)^2))
rf.val.r2 <- cor(y.val.reg, rf.val.pred)^2
cat("Random Forest - Validation RMSE:", round(rf.val.rmse, 2), "| R2:", round(rf.val.r2, 3), "\n")

reg.results <- data.frame(
  Model = c("Linear Regression","Regression Tree","Random Forest"),
  Val_RMSE = round(c(lm.val.rmse, reg.tree.val.rmse, rf.val.rmse), 2),
  Val_R2 = round(c(lm.val.r2, reg.tree.val.r2, rf.val.r2), 3))
cat("\nRegression - Validation Comparison:\n")
print(reg.results)

# Random Forest has lowest RMSE so Linear Regression is selected
# because it directly answers whether catalog engagement independently predicts
# spending after controlling for income - the core question of Business Goal 3
# R2 of 0.851 on validation is strong enough to justify this choice
cat("\nLinear Regression - Test Set:\n")
lm.test.pred <- predict(lm.fit, X.test.reg)
lm.test.rmse <- sqrt(mean((y.test.reg - lm.test.pred)^2))
lm.test.r2 <- cor(y.test.reg, lm.test.pred)^2
cat("RMSE:", round(lm.test.rmse, 2), "| R2:", round(lm.test.r2, 3), "\n")


# TIME SERIES - CATALOG ENGAGEMENT TREND (BP2)

# Monthly avg catalog purchases among new enrollees
# August 2012 is the start - July 2012 had only 2 records (partial month)
monthly <- df %>%
  group_by(EnrollMonth) %>%
  summarise(AvgCatalog = round(mean(NumCatalogPurchases), 2), .groups = "drop") %>%
  filter(EnrollMonth >= "2012-08") %>%
  mutate(Date = as.Date(paste0(EnrollMonth, "-01")))

ts.catalog <- ts(monthly$AvgCatalog, start = c(2012, 8), frequency = 12)

ggplot(monthly, aes(x = Date, y = AvgCatalog)) +
  geom_line(color = "#0F6E56", linewidth = 1) +
  geom_point(color = "#0F6E56", size = 2) +
  geom_smooth(method = "lm", se = TRUE, color = "#D85A30", fill = "#F5C4B3", alpha = 0.25, linewidth = 0.9) +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months") +
  labs(title = "BP2: Avg Catalog Purchases of New Enrollees - Declining Trend",
  x = "Enrollment Month", y = "Avg Catalog Purchases") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 3-month moving average to smooth noise before fitting trend model
ma3 <- ma(ts.catalog, order = 3)
plot(ts.catalog, main = "Actual vs 3-Month Moving Average",
ylab = "Avg Catalog Purchases", col = "#0F6E56", lwd = 1.5)
lines(ma3, col = "#D85A30", lwd = 2)
legend("topright", legend = c("Actual","3-Month MA"), col = c("#0F6E56","#D85A30"), lwd = c(1.5, 2))

# Linear trend model and 6-month forward forecast
trend.model <- tslm(ts.catalog ~ trend)
summary(trend.model)

# checking residuals runs Ljung-Box test and plots ACF to check for autocorrelation
checkresiduals(trend.model)

catalog.forecast <- forecast(trend.model, h = 6, level = 95)
plot(catalog.forecast, main = "BP2: Forecast - Avg Catalog Purchases Jul-Dec 2014",
 ylab = "Avg Catalog Purchases", xlab = "Time")
print(catalog.forecast)


# STEP 12: MODEL PERFORMANCE SUMMARY

cat("\nCLASSIFICATION PERFORMANCE (BP2):\n")
print(class.results)
cat("\nLogistic Regression - Test Set:\n")
print(cm.logit.test$overall["Accuracy"])
print(cm.logit.test$byClass[c("Sensitivity","Specificity")])

cat("\nREGRESSION PERFORMANCE (BP3):\n")
print(reg.results)
cat("Linear Regression - Test Set RMSE:", round(lm.test.rmse, 2), "| R2:", round(lm.test.r2, 3), "\n")

cat("\nCLUSTERING PROFILE (BP1):\n")
print(cluster.profile[, c("Cluster","n","AvgSpend","AvgIncome","AvgCatalog","AvgKidhome","PctHighValue")])

cat("\nTIME SERIES TREND (BP2):\n")
print(summary(trend.model)$coefficients)
print(catalog.forecast)


# AI ASSISTANCE DISCLOSURE
#
# AI assistance was used only for specific visualizations that go beyond what the textbook
# examples demonstrate, specifically charts built to make my business problems directly visible.
#
# 1. Correlation heatmap with numeric labels (Step 6):
#    The textbook shows heatmap() which displays color only, without the actual
#    correlation values printed inside each cell. I needed the numbers visible
#    in the cells so the catalog vs. web contrast could be read directly without
#    switching to a separate table. I used corrplot() and got AI help with the
#    specific arguments - colorRampPalette, addCoef.col, number.cex, and mar -
#    to produce a labeled, readable output tied to Business Problem 3.
#
# 2. Lifecycle stage and web visit quintile bar charts (Step 6):
#    The textbook does not include charts that show spending across household
#    lifecycle stages or across web visit frequency groups. I built these to make
#    Business Problems 1 and 3 directly visible. The ggplot assembly with
#    dollar-formatted bar labels, ordered factor levels on the x-axis, and
#    custom color mapping, all these were done by AI help to put together correctly.
#
# 3. Revenue concentration curve (Step 6):
#    I needed a cumulative chart showing that a small share of consumers
#    generate most of the revenue, to support the business case for targeted
#    retention. This type of chart is not in the textbook examples. AI helped
#    with the cumulative sum calculation and the annotate() layer that reads
#    the percentage directly from the data.
#
# 4. Time series trend line with ggplot (Time Series section):
#    The textbook uses base R plot() for all time series visualizations. I
#    wanted a cleaner chart showing the monthly catalog trend with a fitted
#    line and confidence band using ggplot, with a properly formatted date
#    axis (scale_x_date, date_breaks, angled labels). This ggplot approach
#    for time series date axes is not shown in the textbook, I was looking 
#    at predictive textbook for codes at this time where I couldn't find 
#    relavant codes. So, AI helped assemble the date formatting and geom_smooth overlay correctly.
