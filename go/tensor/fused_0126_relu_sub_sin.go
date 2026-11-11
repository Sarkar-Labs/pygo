package tensor

import "fmt"

func Fused0126ReluSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Sin()
	return r, nil
}
func Fused0126ReluSubSinName() string { return "relu_sub_sin" }
