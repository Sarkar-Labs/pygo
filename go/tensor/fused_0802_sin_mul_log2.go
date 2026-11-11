package tensor

import "fmt"

func Fused0802SinMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.Log2()
	return r, nil
}
func Fused0802SinMulLog2Name() string { return "sin_mul_log2" }
