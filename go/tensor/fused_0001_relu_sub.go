package tensor

import "fmt"

func Fused0001ReluSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	return r, nil
}
func Fused0001ReluSubName() string { return "relu_sub" }
