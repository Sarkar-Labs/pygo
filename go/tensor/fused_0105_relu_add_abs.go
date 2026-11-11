package tensor

import "fmt"

func Fused0105ReluAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Abs()
	return r, nil
}
func Fused0105ReluAddAbsName() string { return "relu_add_abs" }
