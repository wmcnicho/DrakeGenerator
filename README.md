# DrakeGenerator
A project to see if we can use Recurrent Neural Networks to generate bars.

## About
This is a personal project where I'm experimenting with using TensorFlow to build recurrent neural networks to do language modeling/generation. This is a character-level model so the output contains a lot of invalid english words...they sometimes rhyme though! The dataset is all of drakes lyrics uploaded by kaggle user [Juico Bowley](https://www.kaggle.com/juicobowley).

I've implemented a couple different recurrent cells including a "vanilla cell", GRU, and LSTM. GRU seems to generate the most realistic output.

## Usage
Currently this is a bit of a mess you can see an example of this in action by running `custom_bars.py` in a python interpreter.

**Note:** there is some explicit language filtering on the output but some explitives do leak through, generate at your own risk!

## Example output

```
Generating Bars...please wait
Input sequence: [Verse]
1000 character generated sequence:
[Verse]
I don't know what we story to tell me, crish now

[Verse 1: Drake]
To fine since I'm just slide
We won't do not know they're like this play grand alone cless
But we in the fast
'Cause if come about hut somethin'
B-g-Ooh-orla, right, I got high off and make your heart out the life
(One-dripper wito I'm lookin'? (Yeah)
And I said the reason I can never talk at the boing

[Hook]
Take it b****es live

[Verse 2: Voyce]
Issa be like Just we do, it star
Anything other blue
My number Graked, he Ban Tmants Kid

[Bridge]
I’m just tryna find you like
Whatever we don't took what I wastum, scard to the fame, they don't lie tag though, chap and me and chain that new to the studin', from the past though
B****es slow, start the f***ed this s**t masist
Flidal and it's luck, comb
Put out of the dospers go down the Piafurice, what’s hannenive to cames
And that's why women Speaks on you baby
I gotta say that I've been
All why we would really probably get close now) and bottom
Firement told ya Piator Ciry
```
