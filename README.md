# Wassersteain_Gan_financial_time_series
Financial time-series modeling has been a demanding problem since one should consider the convoluted and
stochastic entity of financial processes. Most of the applications of deep neural networks focus on price or volatility
prediction but our goal is to utilize Generative Adversarial Networks (GANs) to build a model which generates realistic
financial time series. As a criterion of how much these generated time series resemble the real data, one can check
whether the model is able to reproduce a set of well-known statistical characteristics of real financial data, or so-called
Stylized Facts. Stylized facts are robust empirical regularities one can observe when studying financial data. They
are key indicators of economic behavior and can provide valuable insights into the underlying mechanisms that drive
the economy. By implementing a Wasserstein Generative Adversarial Network(WGAN) and using different models for
the generator and discriminator, we are able to, up to some degree, reproduce the stylized facts. To further work on
this route, one can consider expanding the database in terms of using more significant features. Deeper networks can
also be implemented hoping to achieve a higher degree of similarity between generated and real financial data.
