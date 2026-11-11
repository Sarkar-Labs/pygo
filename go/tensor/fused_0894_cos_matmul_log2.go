package tensor

import "fmt"

func Fused0894CosMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Log2()
	return r, nil
}
func Fused0894CosMatmulLog2Name() string { return "cos_matmul_log2" }
