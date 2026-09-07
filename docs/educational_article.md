# Five Layers of Math Between "Raw Stock Data" and "Should I Trade?"

If you've studied statistics, linear algebra, and a little machine learning,
you already have the tools to understand how a real quantitative trading
pipeline works. Here's one, broken into five layers — each one a different
branch of math doing a specific job.

## Layer 1: Is this stock trending or bouncing back?

Before predicting anything, you need to characterize *how* a price series
moves. The **Hurst exponent** answers a deceptively simple question: if a
stock went up yesterday, is it more likely to keep going up (trending,
Hurst > 0.5) or reverse (mean-reverting, Hurst < 0.5)? It comes from fractal
geometry — the same math used to measure coastlines. Alongside it,
**autocorrelation** checks whether today's return predicts tomorrow's, and a
rolling **z-score** flags when a stock has moved unusually far from its
recent average, in units of standard deviation.

## Layer 2: How much of this move is just "the market," and how much is unique?

Not every stock move is special — some of it is just the whole market going
up or down together. **Linear regression** separates the two: regress a
stock's daily returns against a handful of broad "factors" (market-wide
returns, small-vs-large-company returns, cheap-vs-expensive-company returns,
momentum). What's left over after removing the market's influence is called
**alpha** — the part of the return that's genuinely about this one stock. The
regression coefficients (**betas**) tell you how sensitive the stock is to
each factor.

## Layer 3: What regime is the market in?

Markets aren't always behaving the same way — sometimes they're calm,
sometimes panicked. A **Markov chain** models this as a small number of
"states" (Bull, Bear, Stagnant) with fixed probabilities of transitioning
between them. The elegant idea, due to mathematician Andrey Markov, is that
you often only need to know where you *are* right now to predict where
you're likely to go next — not the whole history. Multiplying a state vector
by the transition matrix repeatedly gives you a probability forecast several
steps into the future.

## Layer 4: Can a neural network find a pattern in all of this?

A small feedforward neural network takes everything from layers 1–3, plus
raw technical indicators (like RSI and MACD), and learns to classify the
next few days as likely "long," "short," or "flat." This is **supervised
classification** — the same core technique used for image recognition,
applied to numbers describing a stock instead of pixels.

## Layer 5: How much should you actually bet?

Finding a signal is only half the problem — sizing it is the part that ruins
people who get the first four layers right. The **Kelly Criterion**,
developed at Bell Labs in the 1950s from Claude Shannon's information
theory, calculates the mathematically optimal fraction of capital to risk
given your edge and odds. Bet too little and a real edge barely compounds;
bet too much and an ordinary losing streak wipes you out. This layer also
subtracts realistic trading costs before deciding if a signal is worth
acting on.

**The takeaway**: a real trading system isn't one clever idea — it's a chain
of well-understood mathematical tools, each solving one narrow problem, that
only becomes useful when tested rigorously end to end.
