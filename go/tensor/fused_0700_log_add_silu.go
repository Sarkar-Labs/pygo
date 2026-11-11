package tensor

import "fmt"

func Fused0700LogAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.SiLU()
	return r, nil
}
func Fused0700LogAddSiluName() string { return "log_add_silu" }
