package tensor

import "fmt"

func Fused0532SqrtMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.Log2()
	return r, nil
}
func Fused0532SqrtMulLog2Name() string { return "sqrt_mul_log2" }
