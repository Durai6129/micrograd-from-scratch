# micro-grad

> **Break it. Understand it. Build it.**

This project is my attempt to understand neural networks beyond the abstractions.

I started this because I wanted to **document my journey of learning AI**, build something meaningful for my GitHub, and hopefully become more visible to people in the industry.

But there was another reason.

I really like mathematics. That's what pulled me deeper into deep learning and its underlying principles. I had already heard about neural networks and backpropagation, but I mostly understood them at a conceptual level.

I wanted to know:

**What is actually happening underneath all those abstractions?**

So I decided to build a tiny neural-network framework from scratch.

---

## What does "from scratch" mean?

For me, it means being able to **reason about and justify what every part of the code is doing**.

Not just using a framework because it works.

Not treating `.backward()` as magic.

Not blindly calling an optimizer.

I wanted to understand what happens underneath.

That led me to build my own scalar autograd engine and then use it to train a neural network.

---

## What I built

The project gradually evolved from a single scalar value into a working neural network:

```text
Value
  ↓
Neuron
  ↓
Layer
  ↓
MLP
  ↓
Training
  ↓
XOR
```

### `Value`

The foundation is a `Value` object that stores:

* a scalar value
* its gradient
* the values that created it
* the operation that created it
* the logic required to propagate gradients backwards

I implemented operations such as:

* Addition
* Multiplication
* Subtraction
* Division
* Powers
* `tanh`

Each operation also knows how to calculate its local derivative.

This was probably one of the more time-consuming parts for me because I had to manually work through the differentiation for each operator.

---

## The part that finally clicked

Before this project, I knew that gradients and derivatives were involved in neural networks.

But I didn't really understand **why**.

Building the computational graph and implementing backpropagation made gradient descent finally click for me.

The idea that we can use derivatives to determine **how each parameter contributes to the loss**, and then use that information to move the parameters towards a lower loss, completely changed how I looked at neural networks.

The visual explanations from **3Blue1Brown** helped me build the intuition behind this, and videos from **Green Code** also helped me understand the implementation side.

---

## A surprising DSA connection

One of my favourite moments was realizing that I was using something I had previously studied purely as DSA theory:

**Topological sorting.**

I had learned topological sorting as a DSA concept.

Now I was using it to determine the correct order in which nodes of my computational graph should be processed during backpropagation.

That was a genuine dopamine hit.

It's one thing to solve a topological sorting problem for an exam.

It's completely different when you suddenly realize:

> *"Wait... I'm actually using this in a real system."*

That experience made a lot of the theory I'd learned feel more meaningful.

---

## Training an MLP

Once the autograd engine was working, I built:

* `Neuron`
* `Layer`
* `MLP`

The MLP was then trained using gradient descent.

For the first real test, I chose something small but meaningful:

### XOR

```text
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
```

The goal wasn't to build something production-ready.

The goal was to prove that the things I had implemented actually worked together.

And eventually, they did.

---

## My favourite debugging moment

It wasn't completely smooth.

At one point, my loss started behaving randomly instead of gradually decreasing.

The problem?

I had forgotten to **reset the gradients** between training iterations.

Since my engine accumulates gradients, every iteration was carrying information from the previous one.

Once I figured that out, the training started behaving as expected.

And when the network finally produced predictions close to:

```text
0
1
1
0
```

I was genuinely happy.

It felt like I had done something worthwhile that day.

Not because XOR is some impressive AI achievement.

But because **I understood what I had built.**

---

## What I learned

The biggest thing I gained from this project was a real understanding of **gradient descent**.

But beyond that, I learned:

* How computational graphs represent mathematical operations
* How the chain rule becomes backpropagation
* Why gradients need to accumulate
* Why backward propagation needs a correct traversal order
* How derivatives connect parameters to loss
* How a tiny autograd engine can power a neural network
* How much difference it makes to actually implement something instead of only reading about it

And perhaps most importantly:

**I learned how uncomfortable it feels to build something from scratch when you're not used to doing it — and how satisfying it is once the pieces finally start making sense.**

---

## Project structure

```text
micro-grad/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
├── tests/
├── examples/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## What's next?

This project is only the beginning.

My next goal is to go deeper into modern deep learning by implementing a **Transformer architecture / small language model from scratch**.

The idea is to continue the same approach:

**Don't just use the technology. Break it apart and understand what's inside.**

---

## Why I built this

I believe abstraction is useful, but abstraction shouldn't stop us from understanding what lies underneath it.

If we want to build something new, we should first understand the technology that already exists.

> **"If you need to invent something new, first understand the existing technology."**

This project is my first step towards doing exactly that.

**Break it. Understand it. Build it.**

---

## Acknowledgements

A huge thanks to **Andrej Karpathy** and his *Neural Networks: Zero to Hero* playlist.

This project was heavily inspired by and implemented while following that series. His way of breaking down neural networks from the fundamentals made it possible for me to go from *knowing what neural networks are* to actually understanding and implementing the pieces underneath them.

I didn't want to simply reproduce the code. I wanted to understand **why each piece exists, how it works, and what happens when I implement it myself**.

Also, thanks to **3Blue1Brown** for the incredible visual explanations that helped me build the mathematical intuition behind gradients and neural networks, and **Green Code** for the additional implementation perspective.

This project wouldn't have happened without these resources.