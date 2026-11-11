package tensor

import "fmt"

func Fused0804SinMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Log2()
	return r, nil
}
func Fused0804SinMatmulLog2Name() string { return "sin_matmul_log2" }
