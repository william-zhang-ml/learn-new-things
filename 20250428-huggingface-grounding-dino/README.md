So today I paged through some Huggingface examples and also talked to a friend
about a Huggingface thing that they've done. He mentioned using Grounding DINO(text-to-bbox) as a middleman for SAM (prompt-to-segmentation). SAM does not
currently take text prompting so Grounding DINO is a way past the issue.

I've been aware of SAM but Grounding DINO is new to me. So I dedicate this
example to going through the Grounding DINO example code and running some
toy images of electrical components through it just to see what will happen.