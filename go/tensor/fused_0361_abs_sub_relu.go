package tensor

import "fmt"

func Fused0361AbsSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.ReLU()
	return r, nil
}
func Fused0361AbsSubReluName() string { return "abs_sub_relu" }
