package tensor

import "fmt"

func Fused0534SqrtMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Log2()
	return r, nil
}
func Fused0534SqrtMatmulLog2Name() string { return "sqrt_matmul_log2" }
