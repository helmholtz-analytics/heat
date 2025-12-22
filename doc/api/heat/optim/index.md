Module heat.optim
=================
Optimizer module.

It contains data parallel specific optimizers and learning rate schedulers. It also includes all of the
optimizers and learning rate schedulers in the torch namespace

Sub-modules
-----------
* heat.optim.dp_optimizer
* heat.optim.lr_scheduler
* heat.optim.tests
* heat.optim.utils

Classes
-------

`ASGD(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.01, lambd: float = 0.0001, alpha: float = 0.75, t0: float = 1000000.0, weight_decay: float = 0, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False, capturable: bool = False)`
:   Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Adadelta(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 1.0, rho: float = 0.9, eps: float = 1e-06, weight_decay: float = 0, foreach: bool | None = None, *, capturable: bool = False, maximize: bool = False, differentiable: bool = False)`
:   Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Adafactor(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.01, beta2_decay: float = -0.8, eps: tuple[float | None, float] = (None, 0.001), d: float = 1.0, weight_decay: float = 0.0, *, foreach: bool | None = None, maximize: bool = False)`
:   Implements Adafactor algorithm.

    .. math::
        \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \tau
                \text{(}\beta_2\text{ decay)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},    \\
            &\hspace{15mm}      \: \epsilon_1, \epsilon_2 \text{ (epsilons)}, \: d \text{(clipping threshold)}, \\
            &\hspace{15mm}      \: \lambda \text{(weight decay)},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : \: R_0 \leftarrow 0 \text{ (second moment row factor)},       \\
            &\hspace{23mm} \: C_0 \leftarrow 0 \text{ (second moment col factor)},               \\
            &\hspace{23mm} \: \widehat{V}_0 \leftarrow 0 \text{ (second moment for vectors)}     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}G_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}G_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\widehat{\beta}_{2_t} \leftarrow 1 - t^{\tau}                           \\
            &\hspace{5mm}\rho_t         \leftarrow min(lr, \frac{1}{\sqrt{t}})                   \\
            &\hspace{5mm}\alpha_t       \leftarrow max(\epsilon_2,
                \text{RMS}(\theta_{t-1}))\rho_t                                                  \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}    \\
            &\hspace{5mm}\textbf{if} \: \text{dim}(G_t) > 1:                                     \\
            &\hspace{10mm}R_t           \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                (1-\widehat{\beta}_{2_t})(G_t \odot G_t) \cdot 1_m                               \\
            &\hspace{10mm}C_t           \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t)                         \\
            &\hspace{10mm}\widehat{V}_t \leftarrow
                \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{V}_t \leftarrow \widehat{\beta}_{2_t}\widehat{V}_{t-1}+
                (1-\widehat{\beta}_{2_t}) \cdot (G_t \odot G_t)                                  \\
            &\hspace{5mm}U_t            \leftarrow
                \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}                                \\
            &\hspace{5mm}\widehat{U}_t  \leftarrow \frac{U_t}{max(1, \frac{\text{RMS}(U_t)}{d})} \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \alpha_t \widehat{U}_t         \\

            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        \end{aligned}

    For further details regarding the algorithm we refer to `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): unlike other optimizers, Adafactor does not require a
            learning rate, and Shazeer, Noam, and Mitchell Stern do not use lr at all.
            Deviating from the paper, this implementation uses lr for applying weight
            decay and as the maximum value for relative step size rho_t. Note that in
            the paper, a constant of 0.01 is used as the maximum value for relative
            step size, and so we set 0.01 as the default value. (default: 1e-2)
        beta2_decay (float, optional): the decay rate of beta2. beta2 standardly refers
            to the coefficient used for computing the running average of the gradient
            squared. (default: -0.8)
        eps (Tuple[float, float], optional): epsilon1 is the term added to the denominator
            of the update calculation to improve numerical stability. This use of epsilon1
            deviates from the algorithm written in the paper! See note below for more details.
            epsilon2 is the term used to avoid having too small a weight update when applying
            parameter scaling. (default: (None, 1e-3))
        d (float, optional): the clipping threshold, used to avoid larger-than-desired
            updates.
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        foreach (bool, optional): whether foreach implementation of optimizer is used. Note
            that the foreach implementation uses ~ sizeof(params) more peak memory than the
            for-loop version due to the intermediates being a tensorlist vs just one tensor.
            As Adafactor is commonly used when memory is prohibitive, Adafactor will default
            to the slower single tensor for-loop implementation unless this flag is explicitly
            True. This behavior is contrary to other optimizers, which will attempt defaulting
            to foreach on CUDA for faster runtime. (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
    .. Note::
        The implementation of Adafactor subtly differs from Shazeer, Noam, and Mitchell Stern
        and implementations in some other frameworks with its use of learning rate and
        :math:`\epsilon_1`.

        Regarding the learning rate hyperparameter: Shazeer, Noam, and Mitchell Stern do not
        use lr at all, as the stated algorithm uses :math:`\rho_t` and update clipping to
        affect the step size.

        This implementation allows `lr` to influence the maximum value for :math:`\rho_t`:

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(lr, \frac{1}{\sqrt{t}})
            \end{aligned}

        This differs from Shazeer, Noam, and Mitchell Stern, who use a constant of 0.01 as
        the maximum value of :math:`\rho_t`

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(0.01, \frac{1}{\sqrt{t}})
            \end{aligned}

        Shazeer, Noam, and Mitchell Stern do not enforce an opinion on how weight decay should
        be computed, and so we use the learning rate as a coefficient for decoupled weight
        decay, similar to what is suggested in `Decoupled Weight Decay Regularization`_.

        Regarding the use of :math:`\epsilon_1`: The implementation attempts to replicate the
        presumed intention of Shazeer, Noam, and Mitchell Stern to use :math:`\epsilon_1` as
        a stabilizing term when the squared gradient becomes small.

        This stabilization can be written as

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                    (1-\widehat{\beta}_{2_t})(G_t \odot G_t + 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                    (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow
                    \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}        \\
            \end{aligned}

        where the row and column factors of gradient squared :math:`R_t` and :math:`C_t`
        are left alone, and we apply :math:`\epsilon_1` at the final calculation of
        the variance estimate :math:`\widehat{V}_t` and for the update :math:`U_t`.

        This is in contrast to Shazeer, Noam, and Mitchell Stern and other frameworks which
        apply :math:`\epsilon_1` to both row and column factors of the squared gradient, but
        not in the calculations after:

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                            (1-\widehat{\beta}_{2_t})(G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                            (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow \frac{R_t \cdot C_t}{1^\top_n \cdot R_t}                          \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{\sqrt{\widehat{V}_t}}                                            \\
            \end{aligned}


    .. _Adafactor\: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Adagrad(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.01, lr_decay: float = 0, weight_decay: float = 0, initial_accumulator_value: float = 0, eps: float = 1e-10, foreach: bool | None = None, *, maximize: bool = False, differentiable: bool = False, fused: bool | None = None)`
:   Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow \tau                          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the
            sum of squares of gradients (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        fused (bool, optional): whether the fused implementation (CPU only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None). Please note that the fused implementations does not
            support sparse or complex gradients.
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `share_memory(self)`
    :

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Adam(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.001, betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, amsgrad: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False, fused: bool | None = None, decoupled_weight_decay: bool = False)`
:   Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize},  \: \epsilon \text{ (epsilon)}                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: v_0^{max}\leftarrow 0          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm} v_t^{max} \leftarrow \mathrm{max}(v_{t-1}^{max},v_t)                  \\
            &\hspace{10mm}\widehat{v_t} \leftarrow v_t^{max}/\big(1-\beta_2^t \big)              \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): if True, this optimizer is
            equivalent to AdamW and the algorithm will not accumulate weight
            decay in the momentum nor variance. (default: False)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        fused (bool, optional): whether the fused implementation is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation, with fused being theoretically fastest with both
              vertical and horizontal fusion. As such, if the user has not specified either
              flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. Why not fused? Since the fused
              implementation is relatively new, we want to give it sufficient bake-in time.
              To specify fused, pass True for fused. To force running the for-loop
              implementation, pass False for either foreach or fused.
    .. Note::
        A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Descendants

    * torch.optim.adamw.AdamW

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`AdamW(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.001, betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False, *, maximize: bool = False, foreach: bool | None = None, capturable: bool = False, differentiable: bool = False, fused: bool | None = None)`
:   Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: v_0^{max}\leftarrow 0                        \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm} v_t^{max} \leftarrow \mathrm{max}(v_{t-1}^{max},v_t)                  \\
            &\hspace{10mm}\widehat{v_t} \leftarrow v_t^{max}/\big(1-\beta_2^t \big)              \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        fused (bool, optional): whether the fused implementation is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation, with fused being theoretically fastest with both
              vertical and horizontal fusion. As such, if the user has not specified either
              flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. Why not fused? Since the fused
              implementation is relatively new, we want to give it sufficient bake-in time.
              To specify fused, pass True for fused. To force running the for-loop
              implementation, pass False for either foreach or fused.
    .. Note::
        A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    ### Ancestors (in MRO)

    * torch.optim.adam.Adam
    * torch.optim.optimizer.Optimizer

`Adamax(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.002, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, foreach: bool | None = None, *, maximize: bool = False, differentiable: bool = False, capturable: bool = False)`
:   Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`LBFGS(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 1, max_iter: int = 20, max_eval: int | None = None, tolerance_grad: float = 1e-07, tolerance_change: float = 1e-09, history_size: int = 100, line_search_fn: str | None = None)`
:   Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float, optional): learning rate (default: 1)
        max_iter (int, optional): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int, optional): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float, optional): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int, optional): update history size (default: 100).
        line_search_fn (str, optional): either 'strong_wolfe' or None (default: None).

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure)`
    :   Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.

`NAdam(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.002, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, momentum_decay: float = 0.004, decoupled_weight_decay: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False)`
:   Implements NAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)}    \\
            &\hspace{13mm} \: \textit{decoupled\_weight\_decay}, \:\textit{maximize}             \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{15mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}                    \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  0.96^{t \psi} \big)     \\
            &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96^{(t+1)\psi}\big)\\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i)\\[-1.ex]
            & \hspace{11mm} + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain NAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Optimizer(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], defaults: dict[str, typing.Any])`
:   Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    ### Descendants

    * torch.optim.Adafactor
    * torch.optim.adadelta.Adadelta
    * torch.optim.adagrad.Adagrad
    * torch.optim.adam.Adam
    * torch.optim.adamax.Adamax
    * torch.optim.asgd.ASGD
    * torch.optim.lbfgs.LBFGS
    * torch.optim.nadam.NAdam
    * torch.optim.radam.RAdam
    * torch.optim.rmsprop.RMSprop
    * torch.optim.rprop.Rprop
    * torch.optim.sgd.SGD
    * torch.optim.sparse_adam.SparseAdam

    ### Class variables

    `OptimizerPostHook: TypeAlias`
    :

    `OptimizerPreHook: TypeAlias`
    :

    ### Static methods

    `profile_hook_step(func: Callable[~_P, ~R]) ‑> Callable[~_P, ~R]`
    :

    ### Methods

    `add_param_group(self, param_group: dict[str, typing.Any]) ‑> None`
    :   Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.

    `load_state_dict(self, state_dict: dict[str, typing.Any]) ‑> None`
    :   Load the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        .. note::
            The names of the parameters (if they exist under the "param_names" key of each param group
            in :meth:`state_dict`) will not affect the loading process.
            To use the parameters' names for custom cases (such as when the parameters in the loaded state dict
            differ from those initialized in the optimizer),
            a custom ``register_load_state_dict_pre_hook`` should be implemented to adapt the loaded dict
            accordingly.
            If ``param_names`` exist in loaded state dict ``param_groups`` they will be saved and override
            the current names, if present, in the optimizer state. If they do not exist in loaded state dict,
            the optimizer ``param_names`` will remain unchanged.

    `register_load_state_dict_post_hook(self, hook: Callable[[ForwardRef('Optimizer')], None], prepend: bool = False) ‑> torch.utils.hooks.RemovableHandle`
    :   Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `register_load_state_dict_pre_hook(self, hook: Callable[[ForwardRef('Optimizer'), dict[str, Any]], dict[str, Any] | None], prepend: bool = False) ‑> torch.utils.hooks.RemovableHandle`
    :   Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `register_state_dict_post_hook(self, hook: Callable[[ForwardRef('Optimizer'), dict[str, Any]], dict[str, Any] | None], prepend: bool = False) ‑> torch.utils.hooks.RemovableHandle`
    :   Register a state dict post-hook which will be called after :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `register_state_dict_pre_hook(self, hook: Callable[[ForwardRef('Optimizer')], None], prepend: bool = False) ‑> torch.utils.hooks.RemovableHandle`
    :   Register a state dict pre-hook which will be called before :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.
        The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
        The registered hook can be used to perform pre-processing before the ``state_dict``
        call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `register_step_post_hook(self, hook: Callable[[Self, tuple[Any, ...], dict[str, Any]], None]) ‑> torch.utils.hooks.RemovableHandle`
    :   Register an optimizer step post hook which will be called after optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `register_step_pre_hook(self, hook: Callable[[Self, tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]] | None]) ‑> torch.utils.hooks.RemovableHandle`
    :   Register an optimizer step pre hook which will be called before optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

    `state_dict(self) ‑> dict[str, typing.Any]`
    :   Return the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.
            If a param group was initialized with ``named_parameters()`` the names
            content will also be saved in the state dict.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                        'param_names' ['param0']  (optional)
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                        'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
                    }
                ]
            }

    `step(self, closure: Callable[[], float] | None = None) ‑> float | None`
    :   Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

    `zero_grad(self, set_to_none: bool = True) ‑> None`
    :   Reset the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).

`RAdam(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, decoupled_weight_decay: bool = False, *, foreach: bool | None = None, maximize: bool = False, capturable: bool = False, differentiable: bool = False)`
:   Implements RAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weightdecay)}, \:\textit{maximize}                               \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}, \textit{decoupled\_weight\_decay}         \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{12mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{6mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{6mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{12mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{18mm} \theta_t \leftarrow \theta_{t} - \gamma \lambda \theta_{t}            \\
            &\hspace{12mm}\textbf{else}                                                          \\
            &\hspace{18mm} g_t \leftarrow g_t + \lambda \theta_{t}                               \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\[0.1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
      \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.

    This implementation provides an option to use either the original weight_decay implementation as in Adam
    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information
    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.


    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain RAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _author's implementation:
        https://github.com/LiyuanLucasLiu/RAdam
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`RMSprop(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.01, alpha: float = 0.99, eps: float = 1e-08, weight_decay: float = 0, momentum: float = 0, centered: bool = False, capturable: bool = False, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False)`
:   Implements RMSprop algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \alpha \text{ (alpha)}, \: \gamma \text{ (lr)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm}   \lambda \text{ (weight decay)},\: \mu \text{ (momentum)},
                \: centered, \: \epsilon \text{ (epsilon)}                                       \\
            &\textbf{initialize} : v_0 \leftarrow 0 \text{ (square average)}, \:
                \textbf{b}_0 \leftarrow 0 \text{ (buffer)}, \: g^{ave}_0 \leftarrow 0     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}v_t           \leftarrow   \alpha v_{t-1} + (1 - \alpha) g^2_t
                \hspace{8mm}                                                                     \\
            &\hspace{5mm} \tilde{v_t} \leftarrow v_t                                             \\
            &\hspace{5mm}if \: centered                                                          \\
            &\hspace{10mm} g^{ave}_t \leftarrow g^{ave}_{t-1} \alpha + (1-\alpha) g_t            \\
            &\hspace{10mm} \tilde{v_t} \leftarrow \tilde{v_t} -  \big(g^{ave}_{t} \big)^2        \\
            &\hspace{5mm}if \: \mu > 0                                                           \\
            &\hspace{10mm} \textbf{b}_t\leftarrow \mu \textbf{b}_{t-1} +
                g_t/ \big(\sqrt{\tilde{v_t}} +  \epsilon \big)                                   \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} - \gamma \textbf{b}_t                \\
            &\hspace{5mm} else                                                                   \\
            &\hspace{10mm}\theta_t      \leftarrow   \theta_{t-1} -
                \gamma  g_t/ \big(\sqrt{\tilde{v_t}} + \epsilon \big)  \hspace{3mm}              \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `lecture notes <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ by G. Hinton.
    and centered version `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\gamma/(\sqrt{v} + \epsilon)` where :math:`\gamma`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`Rprop(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.01, etas: tuple[float, float] = (0.5, 1.2), step_sizes: tuple[float, float] = (1e-06, 50), *, capturable: bool = False, foreach: bool | None = None, maximize: bool = False, differentiable: bool = False)`
:   Implements the resilient backpropagation algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the paper
    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
        capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`SGD(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.001, momentum: float = 0, dampening: float = 0, weight_decay: float | torch.Tensor = 0, nesterov: bool = False, *, maximize: bool = False, foreach: bool | None = None, differentiable: bool = False, fused: bool | None = None)`
:   Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero. (default: False)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        fused (bool, optional): whether the fused implementation is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation, with fused being theoretically fastest with both
              vertical and horizontal fusion. As such, if the user has not specified either
              flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. Why not fused? Since the fused
              implementation is relatively new, we want to give it sufficient bake-in time.
              To specify fused, pass True for fused. To force running the for-loop
              implementation, pass False for either foreach or fused.


    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.

        Moreover, the initial value of the momentum buffer is set to the
        gradient value at the first step. This is in contrast to some other
        frameworks that initialize it to all zeros.

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.

`SparseAdam(params: Iterable[torch.Tensor] | Iterable[dict[str, typing.Any]] | Iterable[tuple[str, torch.Tensor]], lr: float | torch.Tensor = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, maximize: bool = False)`
:   SparseAdam implements a masked version of the Adam algorithm
    suitable for sparse gradients. Currently, due to implementation constraints (explained
    below), SparseAdam is only intended for a narrow subset of use cases, specifically
    parameters of a dense layout with gradients of a sparse layout. This occurs in a
    special case where the module backwards produces grads already in a sparse layout.
    One example NN module that behaves as such is ``nn.Embedding(sparse=True)``.

    SparseAdam approximates the Adam algorithm by masking out the parameter and moment
    updates corresponding to the zero values in the gradients. Whereas the Adam algorithm
    will update the first moment, the second moment, and the parameters based on all values
    of the gradients, SparseAdam only updates the moments and parameters corresponding
    to the non-zero values of the gradients.

    A simplified way of thinking about the `intended` implementation is as such:

    1. Create a mask of the non-zero values in the sparse gradients. For example,
       if your gradient looks like [0, 5, 0, 0, 9], the mask would be [0, 1, 0, 0, 1].
    2. Apply this mask over the running moments and do computation on only the
       non-zero values.
    3. Apply this mask over the parameters and only apply an update on non-zero values.

    In actuality, we use sparse layout Tensors to optimize this approximation, which means the
    more gradients that are masked by not being materialized, the more performant the optimization.
    Since we rely on using sparse layout tensors, we infer that any materialized value in the
    sparse layout is non-zero and we do NOT actually verify that all values are not zero!
    It is important to not conflate a semantically sparse tensor (a tensor where many
    of its values are zeros) with a sparse layout tensor (a tensor where ``.is_sparse``
    returns ``True``). The SparseAdam approximation is intended for `semantically` sparse
    tensors and the sparse layout is only a implementation detail. A clearer implementation
    would be to use MaskedTensors, but those are experimental.


    .. note::

        If you suspect your gradients are semantically sparse (but do not have sparse
        layout), this variant may not be the best for you. Ideally, you want to avoid
        materializing anything that is suspected to be sparse in the first place, since
        needing to convert all your grads from dense layout to sparse layout may outweigh
        the performance gain. Here, using Adam may be the best alternative, unless you
        can easily rig up your module to output sparse grads similar to
        ``nn.Embedding(sparse=True)``. If you insist on converting your grads, you can do
        so by manually overriding your parameters' ``.grad`` fields with their sparse
        equivalents before calling ``.step()``.


    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure=None)`
    :   Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
