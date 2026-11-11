package tensor

import "fmt"

func Fused1719MeanClone(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("mean_clone: nil operand")
	}
	return x.Mean(0, false), nil
}
func Fused1719MeanCloneName() string { return "mean_clone" }
