import random
class NaiveStatisticModel():
  def __init__(self):
    # dictionary of melody string and its counts over all documents (as integer)
    self.melody_counts = {} 
    # number of all segments, the sum over all counts 
    self.total_segments = 0 
    # dictionaryof melody strings and its hashset of chants that contains
    # this melody
    self.melody_in_chants = {}
    # total number of chants
    self.chant_count = 0


  def predict_segments(self, chants, iterations = 100, 
                       epsilon = 0.05, mu = 5, sigma = 2, 
                       alpha=0.0001, print_each = 5):
    # Do init segmentation, generate model's dictionaries (melody_counts, ...)
    init_segmentation = self.__gaus_rand_segments(chants, mu, sigma)
    # Update chant_count
    self.chant_count = len(chants)
    chant_segmentation = init_segmentation
    for i in range(iterations):
      chant_segmentation = self.__train_iteration(chant_segmentation, epsilon, alpha)
      if i%print_each == 0:
        print("{}. Iteration".format(i))
        top25_melodies = sorted(self.melody_counts, key=self.melody_counts.get, reverse=True)[:30]
        print("\t\t\t", top25_melodies)
        #for topmel in top25_melodies:
        #  print("\t\t\t{}".format(topmel))
    return chant_segmentation
      

  def __gaus_rand_segments(self, chants, mu, sigma):
    rand_segments = []
    for chant_id, chant in enumerate(chants):
      new_chant_segments = []
      i = 0
      while i != len(chant):
        # Find new segment
        new_len = max(int(random.gauss(mu, sigma)), 1)
        k = min(i+new_len, len(chant))
        new_chant_segments.append(chant[i:k])
        last_added_segment = new_chant_segments[-1]
        # Update melody_counts
        if last_added_segment in self.melody_counts:
          self.melody_counts[last_added_segment] += 1
        else:
          self.melody_counts[last_added_segment] = 1
        # Update total_segments count
        self.total_segments += 1
        # Update melody_in_chants
        if last_added_segment in self.melody_in_chants:
          self.melody_in_chants[last_added_segment].add(chant_id)
        else:
          self.melody_in_chants[last_added_segment] = {chant_id}
        # Update i index
        i = k
      rand_segments.append(new_chant_segments)
    return rand_segments

  def __train_iteration(self, segmented_chants, epsilon, alpha):
    new_segmented_chants = []
    join_prev_melody = None
    for chant_id, segments in enumerate(segmented_chants):
      # reset melody_in_chants
      for melody in segments:
        if chant_id in self.melody_in_chants[melody]:
          self.melody_in_chants[melody].remove(chant_id)


      new_segments = []
      for melody in segments:
        self.total_segments -= 1
        self.melody_counts[melody] -= 1

        if join_prev_melody == None:
          # How many documents contains this melody
          chant_frequency = len(self.melody_in_chants[melody])/self.chant_count

          if chant_frequency > epsilon or len(melody) <= 1: 
            # Do nothing, pass the melody to the next stage for joining
            join_prev_melody = melody
          else:
            # Find the best splitting
            max_prob = 0
            left = ""
            right = ""
            for split_point in range(1, len(melody)):
              new_left = melody[:split_point]
              new_right = melody[split_point:]
              left_freq = alpha
              right_freq = alpha
              if new_left in self.melody_counts:
                left_freq += (self.melody_counts[new_left]/self.total_segments)
              if new_right in self.melody_counts:
                right_freq += (self.melody_counts[new_right]/self.total_segments)
              
              if max_prob < left_freq * right_freq:
                max_prob = left_freq * right_freq
                left = new_left
                right = new_right
            # Joining melody with the previous one
            new_segments.append(left)
            new_segments.append(right)
            # Update total_segments count
            self.total_segments += 2
            # Update melody_counts
            if left in self.melody_counts:
              self.melody_counts[left] += 1
            else:
              self.melody_counts[left] = 1
            if right in self.melody_counts:
              self.melody_counts[right] += 1
            else:
              self.melody_counts[right] = 1
        else:
          # Joining melody with the previous one
          new_segments.append(join_prev_melody + melody)
          # Update total_segments count
          self.total_segments += 1
          # Update melody_counts
          if join_prev_melody + melody in self.melody_counts:
            self.melody_counts[join_prev_melody + melody] += 1
          else:
            self.melody_counts[join_prev_melody + melody] = 1
          join_prev_melody = None
          
      # In case of we were about to join melody which is last
      if join_prev_melody != None:
        new_segments.append(join_prev_melody)
        # Update total_segments count
        self.total_segments += 1
        # Update melody_counts
        if join_prev_melody in self.melody_counts:
          self.melody_counts[join_prev_melody] += 1
        else:
          self.melody_counts[join_prev_melody] = 1
        join_prev_melody = None

      # Update melody_in_chants
      for melody in new_segments:
        if melody in self.melody_in_chants:
          self.melody_in_chants[melody].add(chant_id)
        else:
          self.melody_in_chants[melody] = {chant_id}

      new_segmented_chants.append(new_segments)
    return new_segmented_chants