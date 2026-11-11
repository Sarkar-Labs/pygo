package tensor

import "fmt"

func Fused0451SqrtSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.ReLU()
	return r, nil
}
func Fused0451SqrtSubReluName() string { return "sqrt_sub_relu" }
