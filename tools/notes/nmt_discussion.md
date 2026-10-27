### Translation differences

1. **“hi.”**: both models produce `<unk>.` instead of `hola.`<br>
   The larger English vocabulary recognises `hi`, while the smaller one does not.<br>
   That improvement in input coverage does not lead to a correct greeting here.
2. **“i am a good boy.”**: the smaller model gives `soy un buen <unk>.`<br>
   The larger model gives `soy un buen chico.`, which preserves the meaning of “boy”.<br>
   This is a clear improvement; its choice of `chico` is also a valid alternative to the reference's `tipo`.
3. **“may i turn on the radio?”**: both models give `¿puedo <unk> la radio?`<br>
   They retain the request for permission and the radio, but lose the action “turn on”.<br>
   Increasing the vocabulary does not improve this example.
4. **The sentence about Tom seeing Mary again**: the smaller model gives `tom está <unk> a ver a mary.`<br>
   The larger model restores useful words in `tom está deseando volver a mary a ver a mary.`<br>
   Its `deseando volver` captures more of the intended meaning, but it repeats Mary and puts the words in the wrong order.<br>
   The reference is `tom está deseando volver a ver a mary.`
5. **The long sentence about remembering someone from the past**: both models lose the contrast between the person today and the person remembered.<br>
   The smaller model repeats incomplete phrases; the larger model also produces an unfaithful sentence.<br>
   Both still have `occasionally` as an unknown English token.<br>
   The larger vocabulary does not solve this longer sentence.

The larger vocabulary improves coverage and helps some words and phrases, but the translation results are mixed.<br>
It does not resolve every missing word, repetition or change of meaning.<br>
All ten outputs reached their end marker, which shows that generation stopped, not that the translations were correct.<br>
These five fixed examples illustrate particular successes and errors; they do not establish overall sentence accuracy.
