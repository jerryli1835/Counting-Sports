function convertCocoToOpenPose(keypoints, scores) {
    // Combine keypoints and scores into a single array, similar to np.concatenate
    let keypointsInfo = keypoints.map((kp, i) => [...kp, scores[i]]);
  
    // Compute neck joint as the average of the two shoulder points (indexes 5 and 6)
    const neck = keypointsInfo.map((kp) => {
      const leftShoulder = kp[5]; // x, y, score of left shoulder
      const rightShoulder = kp[6]; // x, y, score of right shoulder
      const neckX = (leftShoulder[0] + rightShoulder[0]) / 2;
      const neckY = (leftShoulder[1] + rightShoulder[1]) / 2;
      const neckScore = Math.min(leftShoulder[2], rightShoulder[2]);
      return [neckX, neckY, neckScore];
    });
  
    // Insert the computed neck joint at index 17 in the keypoints array
    keypointsInfo.forEach((kp, i) => {
      kp.splice(17, 0, neck[i]);
    });
  
    // Mapping from COCO keypoint indexes to OpenPose indexes
    const mmposeIdx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3];
    const openposeIdx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17];
  
    // Reorganize keypoints based on the mapping
    keypointsInfo.forEach(kp => {
      openposeIdx.forEach((opIdx, j) => {
        kp[opIdx] = kp[mmposeIdx[j]];
      });
    });
  
    // Extract updated keypoints and scores
    const updatedKeypoints = keypointsInfo.map(kp => kp.map(point => [point[0], point[1]]));
    const updatedScores = keypointsInfo.map(kp => kp.map(point => point[2]));
  
    return { keypoints: updatedKeypoints, scores: updatedScores
  