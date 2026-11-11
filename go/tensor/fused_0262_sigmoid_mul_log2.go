package tensor

import "fmt"

func Fused0262SigmoidMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Log2()
	return r, nil
}
func Fused0262SigmoidMulLog2Name() string { return "sigmoid_mul_log2" }
