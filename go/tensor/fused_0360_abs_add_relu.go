package tensor

import "fmt"

func Fused0360AbsAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Abs()
	r = r.ReLU()
	return r, nil
}
func Fused0360AbsAddReluName() string { return "abs_add_relu" }
