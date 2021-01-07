functions {
 vector softmax_id(vector alpha) {
   vector[num_elements(alpha) + 1] alphac1;
   for (k in 1:num_elements(alpha))
     alphac1[k] = alpha[k];
   alphac1[num_elements(alphac1)] = 0;
   return softmax(alphac1);
  }
}

data {
  int<lower=1> N_guides;
  int<lower=1> N_samples;
  int<lower=1> N_replicates;
  int<lower=0> counts[N_samples,N_guides];

  int<lower=1,upper=N_replicates> replicate_mapping[N_samples];

  int<lower=1> num_params;

  vector[N_guides-1] log_size_factors;

  int<lower=1,upper=num_params> num_params_per_sample[N_samples];
  int<lower=1,upper=num_params> param_indices[sum(num_params_per_sample)];
  int<lower=0,upper=1> use_lambda;
  vector[sum(num_params_per_sample)] lambda_mu;
  vector<lower=0>[sum(num_params_per_sample)] lambda_std;
}

transformed data {
  int cumsum_num_params_per_sample[N_samples+1];
  matrix[N_samples,N_guides-1] zero_matrix = rep_matrix(0,N_samples,N_guides-1);
  cumsum_num_params_per_sample[1] = 0;
  for (i in 1:N_samples) {
    cumsum_num_params_per_sample[i+1] = cumsum_num_params_per_sample[i]+num_params_per_sample[i];
  }
}

parameters {
  matrix[num_params,N_guides-1] beta;
  matrix[num_params,N_guides-1] gamma_rep_raw[N_replicates];
  vector<lower=0>[num_params] s2;
  vector[use_lambda ? sum(num_params_per_sample) : 0] lambda_raw;
}

transformed parameters {
  matrix[N_samples,N_guides-1] alpha = zero_matrix;
  matrix<lower=0,upper=1>[N_samples,N_guides] p;
  vector<lower=0>[num_params] s = sqrt(s2);
  vector[use_lambda ? sum(num_params_per_sample) : 0] lambda;
  for (j in 1:N_samples) {
    if (use_lambda == 1) {
      for (idx in (cumsum_num_params_per_sample[j]+1):cumsum_num_params_per_sample[j+1]) {
        lambda[idx] = lambda_mu[idx]+lambda_std[idx]*lambda_raw[idx];
	alpha[j] += (lambda[idx])*(beta[param_indices[idx]] + s[param_indices[idx]]*gamma_rep_raw[replicate_mapping[j]][param_indices[idx]]);
      }
    } else {
      for (param_idx in param_indices[(cumsum_num_params_per_sample[j]+1):cumsum_num_params_per_sample[j+1]]) {
        alpha[j] += beta[param_idx] + s[param_idx]*gamma_rep_raw[replicate_mapping[j]][param_idx];
      }
    }

    p[j] = softmax_id(alpha[j]'+log_size_factors)';
  }
}

model {
  to_vector(beta) ~ normal(0,2);

  to_vector(lambda_raw) ~ normal(0,1);

  for (i in 1:N_replicates) {
    to_vector(gamma_rep_raw[i]) ~ normal(0,1);
  }

  s2 ~ inv_gamma(3,1);

  for (i in 1:N_samples) {
    counts[i] ~ multinomial(p[i]');
  }
}
