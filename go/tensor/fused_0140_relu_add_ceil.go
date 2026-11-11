package tensor

import "fmt"

func Fused0140ReluAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Ceil()
	return r, nil
}
func Fused0140ReluAddCeilName() string { return "relu_add_ceil" }
