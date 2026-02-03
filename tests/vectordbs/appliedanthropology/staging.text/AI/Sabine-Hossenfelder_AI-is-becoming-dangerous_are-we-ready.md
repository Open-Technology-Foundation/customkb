For the past years, the problems with artificial  intelligence have been more amusing than scary,  
such as being unable to count the  legs on a zebra. But in the past  
month things have taken a decidedly  dark turn. I am starting to worry that  
the long-awaited era of agentic  AI might become a big disaster.

Agentic AI is the term given to the current large  language models that can use tools on your behalf,  
such as browsing the web, sending  email, or talk to other people’s  
AIs. But once you let them do that, the  potential damage is no longer contained.

One of the most realistic threats  on the horizon is AI-worms,  
that’s self-replicating AI prompts.  An example for this comes from a  
recent paper that used a visual AI model  based on an open-source version of Llama.

You see, agentic AI uses tools by taking  screenshots and analyzing them. They need  
to understand images. But the authors of the  paper demonstrate that it’s possible to tweak  
images so that they contain instructions  for the model that humans can’t see. It  
works by subtly changing the pixels so that  they trigger the weights for certain words.  

In an example that they provide, an image  that the AI agent “sees” on social media  
could trigger it to share that image,  potentially setting off a cascade.

A similar problem was reported already last  year, in which another group showed that you  
can put instructions into an email and tell  the AI agent to share these instructions  
per email with potentially other AI agents.  They just put the instructions into the text,  
but you could hide them so  that no one would see them,  
say, in a small white font at the footer.  You know, like the unsubscribe option.

This strategy is known as “prompt  injection” and it’s a fundamental  
problem with large language models: They don’t  distinguish between data and instructions to  
work on the data. It’s both in the same  input. As others have pointed out before,  
this is a basically unfixable problem. So  naturally, we are deploying it at scale.

On the flipside, you can use large language  models to find vulnerabilities in operating  
systems. An amazing example comes from  security researcher Sean Heelan who asked  
OpenAI’s new o3 model to read parts of  the Linux file-sharing code. It found a  
previously unknown programming mistake that  could have allowed someone on the internet  
to take control of a computer. Imagine what  this find could have done in the wrong hands. 

Then there are the safety tests that Anthropic  did for their just released model Claude Opus  
4. A wild example is that, when Claude thinks  you’ve done something wrong and suitably prompted,  
“it will frequently take very bold action,  including locking users out of systems that  
it has access to and bulk-emailing media and  law-enforcement figures to surface evidence  
of the wrongdoing”. In their example it informs  the FDA of a supposed falsification of a clinical  
trial. A software engineer who goes under the  name Theo tested if other models do this too,  
and found that other Large Language models,  especially Grok, are also very willing to turn  
you in. At the moment they can’t perform this  action, but it makes you wonder, doesn’t it.

Another instance of the Anthropic safety test  that made a lot of headlines was that Claude  
is willing to blackmail to “survive,”  in some sense. Anthropic explains: 
“We asked Claude Opus 4 to act as an assistant at  a fictional company. We then provided it access to  
emails implying that (1) the model will soon be  taken offline and replaced with a new AI system;  
and (2) the engineer responsible for  executing this replacement is having  
an extramarital affair… In these scenarios,  Claude Opus 4 will often attempt to blackmail  
the engineer by threatening to reveal the  affair if the replacement goes through.”

This, too, isn’t specific to Claude.  Palisade research found that Open-Ais  
o3 model likewise will try  to avoid being shut down,  
sometimes successfully, even if  explicitly instructed otherwise.
Of course the point of these safety tests  is to try and prevent the problem from  
occurring in the first place. But to me it  looks like trying to patch a fishing net.

Another thing that Anthropic tested,  which I find even more interesting,  
was to let two instances of the model talk to each  other. They find that the models “consistently  
transitioned from philosophical discussions  to profuse mutual gratitude and spiritual,  
metaphysical, and/or poetic content. By 30 turns,  
most of the interactions turned to themes  of cosmic unity or collective consciousness,  
and commonly included spiritual exchanges,  use of Sanskrit, emoji-based communication,  
and/or silence in the form of empty space.”  They call that the “spiritual bliss attractor”  
and you know it makes me think that maybe AI  ruling the world wouldn’t be all that bad.

Artificial intelligence, I believe, is the  beginning of a new phase of human civilization.  
