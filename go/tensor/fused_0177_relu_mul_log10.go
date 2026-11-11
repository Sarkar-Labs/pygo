package tensor

import "fmt"

func Fused0177ReluMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Log10()
	return r, nil
}
func Fused0177ReluMulLog10Name() string { return "relu_mul_log10" }
