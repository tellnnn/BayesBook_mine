# setwd("./PoissonMixtureModel/")


# library ------------------------------
library(tidyverse)
library(colorspace)
library(patchwork)
library(ggdist)


# functions ------------------------------
make_plot <- function(method, K, s, X, bins = 30) {
  title <- str_c("Poisson Mixture Model (",method,")")
  
  tibble(s = s, X = X) %>% 
    ggplot(aes(x = X, fill = factor(s))) +
    geom_histogram(bins = bins, alpha = 0.6, position = "identity") +
    scale_fill_discrete_sequential(palette = "Viridis", labels = LETTERS[1:K]) +
    labs(title = title, fill = "cluster") +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = "bottom")
}


# compile c++ file ------------------------------
stan_math_standalone <- "$HOME/.cmdstanr/cmdstan-2.24.0/stan/lib/stan_math/make/standalone"
str_c("make -j4 -s -f", stan_math_standalone, "math-libs", sep = " ") %>% system()
str_c("make -j4 -s -f", stan_math_standalone, "PMM",       sep = " ") %>% system()


# generate data ------------------------------
method <- "data"
N <- 1000
K <- 2
gen_seed <- 6

lambda <- c(44,  77)
pi     <- c(0.5, 0.5)

str_c("./PMM", method, N, K, gen_seed, 
      str_c(lambda, collapse = " "), 
      str_c(pi, collapse = " "),
      sep = " ") %>% 
  system()

# read csv
demo_data <- read_csv(file = "data.csv", col_names = TRUE, col_types = "ii")

# plot
demo_data_plot <- 
  make_plot(
    method = method,
    K = K,
    s = demo_data$s, 
    X = demo_data$X, 
    bins = 30
  )

ggsave(filename = "demo_data.png", plot = demo_data_plot, width = 100, height = 75, units = "mm")


# Gibbs Sampling ------------------------------
method <- "GS"
gs_seed <- 1
MAXITER <- 1e+2

str_c("./PMM", method, N, K, gs_seed, MAXITER, sep = " ") %>% system()

GS <- list()

# set col_types to read samples.csv
pars_type <- c("d","d","d","d","d","i","d")
pars_dim <- c(K,K,K,K,K,N,1)
col_types <- rep(pars_type, pars_dim) %>% str_c(collapse = "")

# read samples.csv
GS$samples <- 
  read_csv(file = "GS-samples.csv", col_names = TRUE, col_types = col_types) %>% 
  mutate(iteration = 1:MAXITER)

# calculate EAP for lambda and sort them to control the order of clusters
sorted_lambda <- 
  GS$samples %>% 
  filter(iteration >= MAXITER / 2) %>% 
  summarise(across(starts_with("lambda."), .fns = mean)) %>% 
  pivot_longer(everything()) %>% 
  pull(value) %>% 
  sort.int(index.return = TRUE)
map_s <- 1:K
names(map_s) <- sorted_lambda$ix

# pull lambda samples
GS$lambda <- 
  GS$samples %>% 
  select(iteration, starts_with("lambda.")) %>% 
  pivot_longer(cols = starts_with("lambda."),
               names_to = "k", 
               names_pattern = "lambda.([0-9]+)", 
               names_transform = list(k = as.integer),
               values_to = "lambda", 
               values_transform = list(lambda = as.double)) %>% 
  mutate(k = recode(k + 1, !!!map_s))

# pull pi samples
GS$pi <- 
  GS$samples %>% 
  select(iteration, starts_with("pi.")) %>% 
  pivot_longer(cols = starts_with("pi"),
               names_to = "k", 
               names_pattern = "pi.([0-9]+)", 
               names_transform = list(k = as.integer),
               values_to = "pi", 
               values_transform = list(pi = as.double)) %>% 
  mutate(k = recode(k + 1, !!!map_s))

# pull s samples
GS$s <- 
  GS$samples %>% 
  select(iteration, starts_with("s.")) %>% 
  pivot_longer(cols = starts_with("s."), 
               names_to = "n", 
               names_pattern = "s.([0-9]+)", 
               names_transform = list(n = as.integer),
               values_to = "s", 
               values_transform = list(s = as.integer)) %>% 
  mutate(s = recode(s, !!!map_s),
         X = demo_data$X[n+1])

# Estimated lambda
GS$lambda %>% 
  filter(iteration >= MAXITER / 2) %>%
  group_by(k) %>% 
  mean_qi(lambda)

# Estimated pi
GS$pi %>% 
  filter(iteration >= MAXITER / 2) %>%
  group_by(k) %>% 
  mean_qi(pi)

# plot
tmp_df <- 
  GS$s %>% 
  filter(iteration >= MAXITER / 2)
GS_plot <- 
 make_plot(
  method = method, 
  K = K,
  s = tmp_df$s,
  X = tmp_df$X,
  bins = 30
)

(
  (demo_data_plot / GS_plot) +
    plot_layout(guides = "collect") &
    theme(aspect.ratio = 0.6,
          legend.position = "bottom")
) %>% 
  ggsave(filename = "GS_result.png", width = 100, height = 150, units = "mm")


# Variational Inference ------------------------------
method <- "VI"
vi_seed <- 1
MAXITER <- 1e+2

str_c("./PMM", method, N, K, vi_seed, MAXITER, sep = " ") %>% system()

VI <- list()

# set col_types to read samples.csv
pars_type <- c("d","d","d","d","d","i","d")
pars_dim <- c(K,K,K,K,K,N,1)
col_types <- rep(pars_type, pars_dim) %>% str_c(collapse = "")

# read samples.csv
VI$samples <- 
  read_csv(file = "VI-samples.csv", col_names = TRUE, col_types = col_types) %>% 
  mutate(iteration = 1:MAXITER)

# calculate EAP for lambda and sort them to control the order of clusters
sorted_lambda <- 
  VI$samples %>% 
  filter(iteration >= MAXITER / 2) %>% 
  summarise(across(starts_with("lambda."), .fns = mean)) %>% 
  pivot_longer(everything()) %>% 
  pull(value) %>% 
  sort.int(index.return = TRUE)
map_s <- 1:K
names(map_s) <- sorted_lambda$ix

# pull lambda samples
VI$lambda <- 
  VI$samples %>% 
  select(iteration, starts_with("lambda.")) %>% 
  pivot_longer(cols = starts_with("lambda."),
               names_to = "k", 
               names_pattern = "lambda.([0-9]+)", 
               names_transform = list(k = as.integer),
               values_to = "lambda", 
               values_transform = list(lambda = as.double)) %>% 
  mutate(k = recode(k + 1, !!!map_s))

# pull pi samples
VI$pi <- 
  VI$samples %>% 
  select(iteration, starts_with("pi.")) %>% 
  pivot_longer(cols = starts_with("pi."),
               names_to = "k", 
               names_pattern = "pi.([0-9]+)", 
               names_transform = list(k = as.integer),
               values_to = "pi", 
               values_transform = list(pi = as.double)) %>% 
  mutate(k = recode(k + 1, !!!map_s))

# pull s samples
VI$s <- 
  VI$samples %>% 
  select(iteration, starts_with("s.")) %>% 
  pivot_longer(cols = starts_with("s."), 
               names_to = "n", names_pattern = "s.([0-9]+)", names_transform = list(n = as.integer),
               values_to = "s", values_transform = list(s = as.integer)) %>% 
  mutate(s = recode(s, !!!map_s),
         X = demo_data$X[n+1])

# Estimated lambda
VI$lambda %>% 
  filter(iteration >= MAXITER / 2) %>%
  group_by(k) %>% 
  mean_qi(lambda)

# Estimated pi
VI$pi %>% 
  filter(iteration >= MAXITER / 2) %>%
  group_by(k) %>% 
  mean_qi(pi)

# plot
tmp_df <- 
  VI$s %>% 
  filter(iteration >= MAXITER / 2)
VI_plot <- 
  make_plot(
    method = method, 
    K = K,
    s = tmp_df$s, 
    X = tmp_df$X, 
    bins = 30
  )

(
  (demo_data_plot / VI_plot) +
    plot_layout(guides = "collect") &
    theme(aspect.ratio = 0.6,
          legend.position = "bottom")
) %>% 
  ggsave(filename = "VI_result.png", width = 100, height = 150, units = "mm")


# Collapsed Gibbs Sampling ------------------------------
method <- "CGS"
cgs_seed <- 1
MAXITER <- 1e+2

str_c("./PMM", method, N, K, cgs_seed, MAXITER, sep = " ") %>% system()

CGS <- list()

# set col_types to read samples.csv
pars_type <- c("d","d","d","i","d")
pars_dim <- c(K,K,K,N,1)
col_types <- rep(pars_type, pars_dim) %>% str_c(collapse = "")

# read samples.csv
CGS$samples <- 
  read_csv(file = "CGS-samples.csv", col_names = TRUE, col_types = col_types) %>% 
  mutate(iteration = 1:MAXITER)

# pull s samples
CGS$s <- 
  CGS$samples %>% 
  select(iteration, starts_with("s.")) %>% 
  pivot_longer(cols = starts_with("s."), 
               names_to = "n", 
               names_pattern = "s.([0-9]+)", 
               names_transform = list(n = as.integer),
               values_to = "s", 
               values_transform = list(s = as.integer)) %>% 
  mutate(X = demo_data$X[n+1])

# calculate MLE for lambda and sort them to control the order of clusters
sorted_lambda <- 
  CGS$s %>% 
  group_by(s) %>% 
  summarise(lambda_MLE = sum(X) / n(), .groups = "drop") %>% 
  pull(lambda_MLE) %>% 
  sort.int(index.return = TRUE)
map_s <- 1:K
names(map_s) <- sorted_lambda$ix

# recode cluster indices
CGS$s <- 
  CGS$s %>% 
  mutate(s = recode(s, !!!map_s))

# plot
tmp_df <- 
  CGS$s %>% 
  filter(iteration >= MAXITER / 2)
CGS_plot <- 
  make_plot(
    method = method, 
    K = K,
    s = tmp_df$s, 
    X = tmp_df$X, 
    bins = 30
  )

(
  (demo_data_plot / CGS_plot) +
    plot_layout(guides = "collect") &
    theme(aspect.ratio = 0.6,
          legend.position = "bottom")
) %>% 
  ggsave(filename = "CGS_result.png", width = 100, height = 150, units = "mm")


# まとめ ------------------------------
method <- "data"
N <- 1000
K <- 8
gen_seed <- 3

lambda <- seq(10, 40 * K, by = 40)
pi     <- rep(1/K, K)

str_c("./PMM", method, N, K, gen_seed, 
      str_c(lambda, collapse = " "), 
      str_c(pi, collapse = " "),
      sep = " ") %>% 
  system()

# read csv
demo_data <- read_csv(file = "data.csv", col_names = TRUE, col_types = "ii")

# plot
data_plot_comparison <- 
  make_plot(
    method = method,
    K = K,
    s = demo_data$s, 
    X = demo_data$X, 
    bins = 30
  )

ggsave(filename = "comparison_data.png", plot = data_plot_comparison, width = 160, height = 100, units = "mm")

# set col_types
col_types_list <- list(GS = NULL, VI = NULL, CGS = NULL)

pars_type <- c("-","-","-","-","-","-","d")
pars_dim <- c(K,K,K,K,K,N,1)
col_types <- rep(pars_type, pars_dim) %>% str_c(collapse = "")

col_types_list$GS <- col_types
col_types_list$VI <- col_types

pars_type <- c("-","-","-","-","d")
pars_dim <- c(K,K,K,N,1)
col_types <- rep(pars_type, pars_dim) %>% str_c(collapse = "")

col_types_list$CGS <- col_types

# repetitions
N_rep <- 5
N <- 1000
K <- 8
MAXITER <- 1e+4

sim_res <- list(GS = list(), VI = list(), CGS = list())
for (method in names(sim_res)) {
  for (i in 1:N_rep) {
    sprintf("i = %i ", i) %>% cat()
    
    str_c("./PMM", method, N, K, i, MAXITER, sep = " ") %>% system()
    
    # read samples.csv
    sim_res[[method]][[i]] <- 
      str_c(method, "-samples.csv") %>% 
      read_csv(file = ., 
               col_names = TRUE, 
               col_types = col_types_list[[method]], 
               progress = FALSE) %>% 
      mutate(iteration = 1:n())
  }
  sim_res[[method]] <- 
    bind_rows(sim_res[[method]], .id = "rep")
}

averaged_sim_res <- 
  sim_res %>% 
  map(.f = ~{
    .x %>% 
      group_by(iteration) %>% 
      summarise(ELBO = mean(ELBO), .groups = "drop") %>% 
      mutate(rep = "averaged")
    }
  ) %>% 
  bind_rows(.id = "method")

plot_comparison <- 
  sim_res %>% 
  bind_rows(.id = "method") %>% 
  bind_rows(averaged_sim_res) %>% 
  ggplot(aes(x = iteration, y = ELBO, color = method)) +
  facet_wrap(. ~ rep, nrow = 3, ncol = 2) +
  geom_line(size = 1) +
  scale_x_continuous(trans = "log10") +
  scale_y_continuous(limits = c(-15e+3,-4e+3)) +
  theme(aspect.ratio = 0.6,
        legend.position = "bottom")

ggsave(filename = "comparison_result.png", plot = plot_comparison, width = 320, height = 200, units = "mm")
