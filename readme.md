# SDE Diffusion

## 1. Theory

### 1.1 forward

$$
\mathbf{d}x = f(x, t)\mathbf{d}t + g(t)\mathbf{d}w
$$

we call $f$ the drift term, and $g$ the diffusion term.

we use marginal prob to add noise at one time

$$
q_t(x_t|x_0) = N(x(t); \gamma_tx_0, \sigma_t^2I)
$$

$\gamma_t, \sigma_t^2$ has close form solution w.r.t $f, g$

### 1.2 reverse

$$
\mathbf{d}x = [f(x, t) - g(t)^2  \nabla_{x} logP_t(x))]\mathbf{d}t + g(t)\mathbf{d}w
$$

discrete form

$$
x_t - x_{t-\Delta t} = [f(x, t) - g(t)^2  \Theta ]\Delta t+ g(t)\sqrt{\Delta t}\mathbf z\\  x_{t-\Delta t} = x_t - [f(x, t) - g(t)^2  \Theta ]\Delta t+ g(t)\sqrt{\Delta t}\mathbf z
$$

since $z$ is the gaussian noise, plus and minus has the same effect.

the only unknown variable in reverse process is $\nabla_{x}logP_t(x)$, use $\nabla_x logP(x_t|x_0)$  to subsitude $\nabla_x logP(x_t)$ , and use nn $\Theta$ to model  $\nabla_x logP(x_t|x_0)$ . note it is a gaussian distribution.

$$
log \nabla_{x} P(x_t|x_0) = \nabla_{x} -\frac{(x_t - \gamma_t x_0)^2}{2\sigma^2} = - \frac{x_t - \gamma_t x_0}{\sigma^2} = - \frac{x_0 + \sigma_t \epsilon - \gamma_t x_0}{\sigma^2} = -\frac{\epsilon}{\sigma_t}
$$

so the model’s output should be  $-\frac{\epsilon}{\sigma_t}$.

## 2. case study: VP

$$
\mathbf{d}x =-\frac{1}{2}\beta(t)x(t)\mathbf{d}t + \sqrt{\beta(t)}\mathbf{d}w
$$

marginal prob for add noise

$$
q_t(x_t|x_0) = N(x(t);\gamma_tx_0, \sigma_t^2I), \gamma_t=e^{-\frac{1}{2}\int_0^t \beta(s)ds}, \sigma_t^2=1-e^{-\int_0^t \beta(s)ds}
$$

reverse process for sampling

$$
x_{t-\Delta t} = x_t - [-\frac{1}{2}\beta(t)x(t) - \beta(t) \Theta ]\Delta t+ \sqrt{\beta(t)}\sqrt{\Delta t}\mathbf z
$$

## 3. sampled image from this code
![img](assets/output.png)

## 4. Implementation Notes

1. in sampling, remember to clip the output.
2. in loss function, the model’s output should be  $-\frac{\epsilon}{\sigma_t}$. and we use

```python
# use sum in c, h, w dim, equal to scale learning rate
train_loss = torch.mean(torch.sum((pred + noise) ** 2, dim=[1, 2, 3]))
```

instead of mse loss, actually they are the same expect the scaling learning rate effect.
