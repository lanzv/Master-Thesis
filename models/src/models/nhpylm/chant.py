from src.models.nhpylm.definitions import BOS_CHAR, BOS, EOS_CHAR, EOS

class Chant:
    """
    This struct holds everything that represents a chant, including the raw string that constitute the chant,
    and the potential segmentation of the chant, if it is segmented,
    either via a preexisting segmentation or via running the forward-filtering-backward-sampling segmentation algorithm.
    """
    def __init__(self, chant_string: str):
        self.chant_string: str = chant_string # UTF32String
        # The individual characters that make up this chant
        self.characters: list = [*chant_string] # OffsetVector{Char}
        self.num_segments: int = 4
        # The length of the segments within this chant.
        self.segment_lengths: list = [0 for _ in range(len(self.characters) + 3)] # OffsetVector{Int}
        self.segment_lengths[0] = 1
        self.segment_lengths[1] = 1
        self.segment_lengths[2] = len(self.characters)
        self.segment_lengths[3] = 1
        self.segment_begin_positions: list = [0 for _ in range(len(self.characters) + 3)] # OffsetVector{Int}
        self.segment_begin_positions[0] = 0
        self.segment_begin_positions[1] = 0
        self.segment_begin_positions[2] = 0
        self.segment_begin_positions[3] = len(self.characters)
        # Indicates whether the chant contains the true segmentation already.
        self.supervised: bool = False
        # The corresponding integer representations of the words. This includes both bos (2) and eos (1)
        # Because `hash` returns UInt, the contents also need to be UInt.
        self.word_ids: list = [0 for _ in range(len(self.characters) + 3)] #OffsetVector{UInt}
        self.word_ids[0] = BOS
        self.word_ids[1] = BOS
        self.word_ids[2] = self.get_substr_word_id(0, len(chant_string) - 1)
        self.word_ids[3] = EOS
        # The string that makes up the chant
        # Note how the `UTF32String` type is used here: In Julia, the indices of the default `String` type are byte indices, not real character indices. For example:
        # ```julia
        #> a = "我们"
        #> a[1]
        #'我': Unicode U+6211 (category Lo: Letter, other)
        #> a[2]
        #ERROR: UTF32StringIndexError("我们", 2)
        #> a[4]
        #'们': Unicode U+4eec (category Lo: Letter, other)
        #```
        #However, in this program we need to constantly access individual characters in the string directly by their indices. Therefore, UTF32String, which always stores its characters in a fixed-width fashion, similar to the `wstring` type in C++, is used.




    def chant(chant_string: str, supervised: bool) -> "Chant":
        chant = Chant(chant_string)
        chant.supervised = supervised
        return chant

    def length(self) -> int:
        return len(self.chant_string)

    def get_num_segments_without_sepcial_tokens(self) -> int:
        return self.num_segments - 3

    def get_nth_segment_length(self, n: int) -> int:
        if not (n <= self.num_segments):
            raise Exception("n > self.num_segments")
        return self.segment_lengths[n]

    def get_nth_word_id(self, n: int):
        if not (n <= self.num_segments):
            raise Exception("n > self.num_segments")
        return self.word_ids[n]

    def get_substr_word_id(self, start_index: int, end_index: int):
        """
        Or just use the built-in hash method if we're to keep the original structure. I'm pretty sure that all the words in a language is not going to break the hashing process.
        Get the word id of the substring with start_index and end_index. Note that in Julia the end_index is inclusive.
        Note that the `hash` method returns UInt! This makes sense because a 2-fold increase in potential hash values can actually help a lot.
        """
        # Let me put +1 on everything involving chant_string since I can't seem to change its indexing easily
        return hash(self.chant_string[int(start_index):int(end_index)+1])

    def get_substr_word_string(self, start_index, end_index) -> str:
        # Let me put +1 on everything involving chant_string since I can't seem to change its indexing easily
        return self.chant_string[start_index:end_index+1]

    def get_nth_word_string(self, n: int) -> str:
        # The last segment is <EOS>
        if not (n < self.num_segments):
            raise Exception("n >= self.num_segments")
        # TODO: This is all hard-coded. We'll need to change them if we're to support bigrams.
        # Can't we make the code a bit more generic? Don't think it would be that hard eh.
        # There are two BOS in the beginning if we use 3-gram.
        if n < 2:
            return "<BOS>"
        else:
            if not n < self.num_segments - 1:
                raise Exception("n >= self.num_segments-1")
            start_position = self.segment_begin_positions[n]
            # OK I see. Somehow the "end_position" didn't - 1. What's this black magic?
            end_position = start_position + self.segment_lengths[n]
            # println("The string length is $(length(s)), the start position is $(start_position), the end_position is $(end_position)")
            # Now we have to + 1 because UTF32String cannot be 0-indexed.
            return self.chant_string[int(start_position):int(end_position)]

    def show(self):
        segments = []
        for index in range(2, self.num_segments-1):
            segments.append(self.get_nth_word_string(index))
        print(segments)

    def split_chant_with_num_segments(self, segment_lengths: list, actual_num_segments: int):
        num_segments_without_special_tokens = actual_num_segments
        cur_start = 0
        sum_length = 0
        index = 0
        while index < num_segments_without_special_tokens:
            if not segment_lengths[index] > 0:
                raise Exception("segment_lengths[index] <= 0")
            sum_length += segment_lengths[index]

            cur_length = segment_lengths[index]

            # + 2 because the first two tokens are BOS.
            self.segment_lengths[index + 2] = cur_length
            self.word_ids[index + 2] = self.get_substr_word_id(cur_start, cur_start + cur_length - 1)
            self.segment_begin_positions[index + 2] = cur_start
            cur_start += cur_length
            index += 1
        # println("sum_length is $sum_length, length of chant_string is $(length(chant.chant_string)), chant is $(chant.chant_string)")
        if not sum_length == len(self.chant_string):
            raise Exception("sum_length != len(self.chant_string)")
        # Also need to take care of EOS now that the actual string ended.
        self.segment_lengths[index + 2] = 1
        self.word_ids[index + 2] = EOS
        # So the EOS token is considered to be a part of the last word?
        self.segment_begin_positions[index + 2] = self.segment_begin_positions[index + 1]
        index += 1
        # If those values were set previously, set them again to 0

        while index < len(self.chant_string):
            self.segment_lengths[index + 2] = 0
            self.segment_begin_positions[index + 2] = 0
            index += 1
        self.num_segments = num_segments_without_special_tokens + 3
        # There doesn't seem to be any problem in this method.
        # println("In split_chant, chant is $chant, chant.num_segments = $(chant.num_segments), chant.segment_lengths is $(chant.segment_lengths)")


    def split_chant(self, segment_lengths: list):
        """
        This method is to split the chant using an already calculated segment_lengths vector, which contains the lengths of each segment.
        Note that the segment_lengths array is without containing any BOS or EOS tokens.
        """
        num_segments_without_special_tokens = len(segment_lengths)
        self.split_chant_with_num_segments(segment_lengths, num_segments_without_special_tokens)